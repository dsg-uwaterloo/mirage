#include <faiss/Index.h>

#include <faiss/impl/MIRAGE.h>
#include <faiss/IndexHNSW.h>
namespace faiss {

using idx_t = faiss::idx_t;

struct IndexMirage : faiss::Index {
    bool own_fields;
    faiss::Index* storage;
    faiss::IndexHNSWFlat hierarchy;
    bool verbose;
    Mirage mirage;

    explicit IndexMirage(int d = 0, int M = 32,
                             faiss::MetricType metric = faiss::METRIC_L2);
    explicit IndexMirage(Index* storage, int M = 32);

    ~IndexMirage() override;

    void add(idx_t n, const float* x) override;

    void train(idx_t n, const float* x) override;
    void search(idx_t n, const float* x, idx_t k, float* distances,
                idx_t* labels,
                const faiss::SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;
    void convert_to_mirage(const Mirage& mirage, faiss::IndexHNSW& hierarchy, faiss::DistanceComputer& qdis, const float* x);
    void add_levels(const Mirage& mirage, faiss::IndexHNSW& hierarchy, const float* x);
    void add_inserts(IndexHNSW& index_hnsw,size_t n0,size_t n,const float* x,bool verbose,bool preset_levels);
    void reset() override;
};

}  // namespace faiss