#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/MIRAGE.h>

#include <iostream>

namespace faiss {

void gen_random(std::mt19937& rng, int* addr, const int size, const int N) {

    for (int i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (int i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    int off = rng() % N;
    for (int i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }

}

Mirage::Mirage(const int d) : d(d) {}

Mirage::~Mirage() {}
void Mirage::init_graph(faiss::DistanceComputer& qdis) {

    graph.reserve(ntotal);
    {
        std::mt19937 rng(random_seed * 6007);
        for (int i = 0; i < ntotal; i++) {
            graph.push_back(faiss::nndescent::Nhood(L, S, rng, (int)ntotal));
        }
    }

#pragma omp parallel
    {
        std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
        for (int i = 0; i < ntotal; i++) {
            std::vector<int> tmp(S);

            gen_random(rng, tmp.data(), S, ntotal);

            for (int j = 0; j < S; j++) {
                int id = tmp[j];
                if (id == i) continue;
                float dist = qdis.symmetric_dis(i, id);

                graph[i].pool.push_back(
                        faiss::nndescent::Neighbor(id, dist, true));
            }
            std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
            graph[i].pool.reserve(L);
        }
    }
}

void Mirage::insert_nn(int id, int nn_id, float distance, bool flag) {
    auto& nhood = graph[id];
    {
        std::lock_guard<std::mutex> guard(nhood.lock);
        nhood.pool.emplace_back(nn_id, distance, flag);
    }
}

void Mirage::update(faiss::DistanceComputer& qdis) {

#pragma omp parallel for schedule(dynamic, 256)
   for (int u = 0; u < ntotal; ++u) {
       auto& nhood = graph[u];

       // Swap the pool with a local temporary to reduce lock duration.
       std::vector<faiss::nndescent::Neighbor> old_pool;
       {
           std::lock_guard<std::mutex> guard(nhood.lock);
           old_pool.swap(nhood.pool);
       }

       // Remove duplicates: sort and unique.
       std::sort(old_pool.begin(), old_pool.end());
       old_pool.erase(
               std::unique(old_pool.begin(), old_pool.end(),
                           [](const faiss::nndescent::Neighbor& a,
                              const faiss::nndescent::Neighbor& b) {
                               return a.id == b.id;
                           }),
               old_pool.end());

       // Prepare new pool and auxiliary set for fast duplicate checks.
       std::vector<faiss::nndescent::Neighbor> new_pool;
       new_pool.reserve(old_pool.size());
       std::unordered_set<int> new_ids;

       // Process each neighbor from the old pool.
       for (const auto& nn : old_pool) {
           // Skip if already added.
           if (new_ids.find(nn.id) != new_ids.end())
               continue;

           bool ok = true;
           // Check against neighbors already in the new pool.
           for (const auto& other_nn : new_pool) {
               // If both flags are false, skip distance check.
               if (!nn.flag && !other_nn.flag)
                   continue;

               // Compute distance between the two candidates.
               float distance = qdis.symmetric_dis(nn.id, other_nn.id);
               if (distance < nn.distance) {
                   ok = false;
                   // Insert a reverse edge.
                   insert_nn(other_nn.id, nn.id, distance, true);
                   break;
               }
           }
           if (ok) {
               new_pool.push_back(nn);
               new_ids.insert(nn.id);
           }
       }

       // Mark all new neighbors as processed.
       for (auto& nn : new_pool)
           nn.flag = false;

       // Reassign the updated pool.
       {
           std::lock_guard<std::mutex> guard(nhood.lock);
           nhood.pool = std::move(new_pool);
       }
   }
}

void Mirage::add_reverse_edges() {
    std::vector<std::vector<faiss::nndescent::Neighbor>> reverse_pools(ntotal);
int num_reverse_edges = 96;
#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        for (auto&& nn : graph[u].pool) {
            std::lock_guard<std::mutex> guard(graph[nn.id].lock);
            reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag);
        }
    }

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        auto& pool = graph[u].pool;
        for (auto&& nn : pool) {
            nn.flag = true;
        }
        auto& rpool = reverse_pools[u];
        rpool.insert(rpool.end(), pool.begin(), pool.end());
        pool.clear();
        std::sort(rpool.begin(), rpool.end());
        rpool.erase(std::unique(rpool.begin(), rpool.end(),
                                [](faiss::nndescent::Neighbor& a,
                                   faiss::nndescent::Neighbor& b) {
                                    return a.id == b.id;
                                }),
                    rpool.end());
        if (rpool.size() > num_reverse_edges) {
            rpool.resize(num_reverse_edges);
        }
    }

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        for (auto&& nn : reverse_pools[u]) {
            std::lock_guard<std::mutex> guard(graph[nn.id].lock);
            graph[nn.id].pool.emplace_back(u, nn.distance, nn.flag);
        }
    }

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        auto& pool = graph[u].pool;
        std::sort(pool.begin(), pool.end());
        if (pool.size() > num_reverse_edges) {
            pool.resize(num_reverse_edges);
        }
    }
}

void Mirage::build(faiss::DistanceComputer& qdis, const int n,
                   bool verbose) {
    if (verbose) {
        printf("Parameters: S=%d, R=%d, Iter=%d\n", S, R, iter);
    }

    ntotal = n;
    init_graph(qdis);

    for (int r = 0; r < R; ++r) {
        if (verbose) {
            std::cout << "R " << r << " : " << std::flush;
        }
        for (int it = 0; it < iter; ++it) {
            update(qdis);
            if (verbose) {
                std::cout << "#" << std::flush;
            }
        }

        if (r != R - 1) {
            add_reverse_edges();
        }

        if (verbose) {
            printf("\n");
        }
    }

#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
        auto& pool = graph[u].pool;
        std::sort(pool.begin(), pool.end());
        pool.erase(std::unique(pool.begin(), pool.end(),
                               [](faiss::nndescent::Neighbor& a,
                                  faiss::nndescent::Neighbor& b) {
                                   return a.id == b.id;
                               }),
                   pool.end());
    }

    offsets.resize(ntotal + 1);
    offsets[0] = 0;
    for (int u = 0; u < ntotal; ++u) {
        offsets[u + 1] = offsets[u] + graph[u].pool.size();
    }

    final_graph.resize(offsets.back(), -1);
#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
        auto& pool = graph[u].pool;
        int offset = offsets[u];
        for (int i = 0; i < pool.size(); ++i) {
            final_graph[offset + i] = pool[i].id;
        }
    }
    std::vector<faiss::nndescent::Nhood>().swap(graph);

    has_built = true;
}

void Mirage::reset() {
    has_built = false;
    ntotal = 0;
    final_graph.resize(0);
    offsets.resize(0);
}

}  // namespace faiss