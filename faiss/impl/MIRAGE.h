#include <faiss/impl/NNDescent.h>

#include <vector>

namespace faiss {

struct Mirage {
    using storage_idx_t = int;

    using KNNGraph = std::vector<faiss::nndescent::Nhood>;

    explicit Mirage(const int d);

    ~Mirage();

    void build(faiss::DistanceComputer& qdis, const int n, bool verbose);

    void reset();

    /// Initialize the KNN graph randomly
    void init_graph(faiss::DistanceComputer& qdis);

    void update(faiss::DistanceComputer& qdis);
    void add_reverse_edges();

    void insert_nn(int id, int nn_id, float distance, bool flag);

    bool has_built = false;

    int R = 4;
    int iter = 15;
    int S = 16;

    int random_seed = 2021;  // random seed for generators

    int d;  // dim
    int L = 8;  // initial size of memory allocation

    int ntotal = 0;

    KNNGraph graph;
    std::vector<int> final_graph;
    std::vector<int> offsets;
};

}  // namespace faiss