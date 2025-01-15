/**
* This implementation is heavily based on faiss::IndexNNDescent.cpp,
 * faiss:: IndexHNSW.cpp and rnndescent https://github.com/mti-lab/rnn-descent
*/

// -*- c++ -*-

#include <omp.h>
#include <faiss/IndexMirage.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <unordered_set>

#ifdef __SSE__
#endif

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n,
          FINTEGER* k, const float* alpha, const float* a, FINTEGER* lda,
          const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);
}

namespace faiss {

using namespace faiss;

using storage_idx_t = NNDescent::storage_idx_t;

/**************************************************************
* add / search blocks of descriptors
**************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
  distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
   /// owned by this
   DistanceComputer* basedis;

   explicit NegativeDistanceComputer(DistanceComputer* basedis)
           : basedis(basedis) {}

   void set_query(const float* x) override { basedis->set_query(x); }

   /// compute distance of vector i to current query
   float operator()(idx_t i) override { return -(*basedis)(i); }

   /// compute distance between two stored vectors
   float symmetric_dis(idx_t i, idx_t j) override {
       return -basedis->symmetric_dis(i, j);
   }

   ~NegativeDistanceComputer() override { delete basedis; }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
   if (is_similarity_metric(storage->metric_type)) {
       return new NegativeDistanceComputer(storage->get_distance_computer());
   } else {
       return storage->get_distance_computer();
   }
}

}  // namespace

/**************************************************************
* IndexMirage implementation
**************************************************************/

IndexMirage::IndexMirage(int d, int K, MetricType metric)
       : Index(d, metric), mirage(d), own_fields(false), storage(nullptr) {
   // the default storage is IndexFlat
   storage = new IndexFlat(d, metric);
   own_fields = true;
}

IndexMirage::IndexMirage(Index* storage, int K)
       : Index(storage->d, storage->metric_type),
         mirage(storage->d),
         own_fields(false),
         storage(storage) {}

IndexMirage::~IndexMirage() {
   if (own_fields) {
       delete storage;
   }
}

void IndexMirage::train(idx_t n, const float* x) {
   FAISS_THROW_IF_NOT_MSG(storage,
                          "Please use IndexNNDescentFlat (or variants) "
                          "instead of IndexNNDescent directly");
   // nndescent structure does not require training
   storage->train(n, x);
   is_trained = true;
}




void IndexMirage::search(idx_t n, const float* x, idx_t k, float* distances,
                            idx_t* labels,
                            const SearchParameters* params) const {
   FAISS_THROW_IF_NOT_MSG(!params,
                          "search params not supported for this index");
   FAISS_THROW_IF_NOT(storage);


   hierarchy.search(n, x, k, distances, labels);
}

void IndexMirage::add(idx_t n, const float* x) {
    int n0 = hierarchy.ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    if (n0 != 0) {
        printf("Incremental Inserts Detected\n");
        hierarchy.storage = storage;
        add_inserts(hierarchy, n0, n, x, verbose, hierarchy.hnsw.levels.size() == ntotal);

    }



   std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
   mirage.build(*dis, ntotal, verbose);

   hierarchy.ntotal = n;
   hierarchy.storage = storage;
   hierarchy.d = d;


   convert_to_mirage(mirage, hierarchy, *dis, x);

}

void IndexMirage::convert_to_mirage(const faiss::Mirage &mirage, faiss::IndexHNSW& hierarchy,
                                                     faiss::DistanceComputer &qdis, const float* x) {

   int n = mirage.ntotal;
   int k = mirage.S;
   add_levels(mirage, hierarchy, x); // add hierarchy first

   std::vector<float> D(n * k, 0.0f);
   std::vector<idx_t> I(n * k, -1);


   // Fill D and I from the final_graph
   // Temporary storage for the query vector
   std::vector<float> query_vector(storage->d);

   for (int u = 0; u < n; ++u) {
       int offset = mirage.offsets[u];
       int degree = mirage.offsets[u + 1] - offset;

       storage->reconstruct(u, query_vector.data());
       qdis.set_query(query_vector.data());

       for (int i = 0; i < degree && i < k; ++i) { // rememeber && i<k
           int neighbor_id = mirage.final_graph[offset + i];
           if (neighbor_id != -1) {
               I[u * k + i] = neighbor_id;
               D[u * k + i] = qdis(neighbor_id);
           }
       }
   }
   hierarchy.init_level_0_from_knngraph(k, D.data(), I.data()); // add Layer 0


}

void IndexMirage::add_levels(const faiss::Mirage &mirage, faiss::IndexHNSW &hierarchy, const float *x) {


   int n = mirage.ntotal;
   size_t d = hierarchy.d;
   HNSW& hnsw = hierarchy.hnsw;
   hnsw.efConstruction = 1024;
   size_t ntotal = n; // change for inserts

   if (n == 0) {
       return;
   }

   int max_level = hierarchy.hnsw.prepare_level_tab_mirage(n, false); // change with inserts

   std::vector<omp_lock_t> locks(ntotal);
   for (int i = 0; i < ntotal; i++)
       omp_init_lock(&locks[i]);

   // add vectors from highest to lowest level
   std::vector<int> hist;
   std::vector<int> order(n);

   { // make buckets with vectors of the same level

       // build histogram
       for (int i = 0; i < n; i++) {
           storage_idx_t pt_id = i; // change for inserts
           int pt_level = hnsw.levels[pt_id] - 1;
           while (pt_level >= hist.size())
               hist.push_back(0);
           hist[pt_level]++;
       }

       // accumulate
       std::vector<int> offsets(hist.size() + 1, 0);
       for (int i = 0; i < hist.size() - 1; i++) {
           offsets[i + 1] = offsets[i] + hist[i];
       }

       // bucket sort
       for (int i = 0; i < n; i++) {
           storage_idx_t pt_id = i; // change for inserts
           int pt_level = hnsw.levels[pt_id] - 1;
           order[offsets[pt_level]++] = pt_id;
       }
   }

   idx_t check_period = InterruptCallback::get_period_hint(
           max_level * hierarchy.d * hnsw.efConstruction);

   { // perform add
       RandomGenerator rng2(789);

       int i1 = n;

       for (int pt_level = hist.size() - 1; pt_level > 0; pt_level--) { // change for inserts

           int i0 = i1 - hist[pt_level];

           if (verbose) {
               printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
           }

           // random permutation to get rid of dataset order bias
           for (int j = i0; j < i1; j++)
               std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

           bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
           {
               VisitedTable vt(ntotal);

               std::unique_ptr<DistanceComputer> dis(storage_distance_computer(hierarchy.storage));

               //std::unique_ptr<DistanceComputer> qdis(storage_distance_computer(storage));
               int prev_display =
                       verbose && omp_get_thread_num() == 0 ? 0 : -1;
               size_t counter = 0;

               // here we should do schedule(dynamic) but this segfaults for
               // some versions of LLVM. The performance impact should not be
               // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
               for (int i = i0; i < i1; i++) {
                   storage_idx_t pt_id = order[i];
                   if(x == NULL){
                       printf("xis null\n");
                   }
                   dis->set_query(x + (pt_id) * d);

                   // cannot break
                   if (interrupt) {
                       continue;
                   }

                   hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                   if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                       prev_display = i - i0;
                       printf("  %d / %d\r", i - i0, i1 - i0);
                       fflush(stdout);
                   }
                   if (counter % check_period == 0) {
                       if (InterruptCallback::is_interrupted()) {
                           interrupt = true;
                       }
                   }
                   counter++;
               }
           }
           if (interrupt) {
               FAISS_THROW_MSG("computation interrupted");
           }
           i1 = i0;
       }
       // FAISS_ASSERT(i1 == 0);
   }

   for (int i = 0; i < ntotal; i++) {
       omp_destroy_lock(&locks[i]);
   }

}
// inserts done in the same manner as HNSW
void IndexMirage::add_inserts(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("add_inserts: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hnsw.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}





void IndexMirage::reset() {
   mirage.reset();
   storage->reset();
   ntotal = 0;
}

void IndexMirage::reconstruct(idx_t key, float* recons) const {
   storage->reconstruct(key, recons);
}

}  // namespace faiss
