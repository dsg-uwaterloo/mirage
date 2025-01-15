# MIRAGE
MIRAGE-ANNS is a state-of-the-art vector index designed for fast construction and superior search

For Implementation details please look at:
```
faiss/faiss/IndexMIRAGE.h
faiss/faiss/IndexMIRAGE.cpp
faiss/faiss/impl/MIRAGE.cpp
faiss/faiss/impl/MIRAGE.h
```

This implementation of the MIRAGE index is built on [**The FAISS Library**](https://github.com/facebookresearch/faiss) in C++.

## Compilation
```
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
make -C build -j faiss
make -C build demo_mirage
./build/demos/demo_mirage

```


## Example Usage (Please look at the demos/demo_mirage.cpp for a full example)
1) Initialize the index
```
// dimension of the vectors to index
int d = 128;

// size of the data we plan to index
size_t nb = 1000000;

// make the index object
faiss::IndexMirage index(d);
index.mirage.S = 32;
index.mirage.R = 4;
index.mirage.iter = 15;
index.verbose = true;
```
2) Construct the index
```
index.add(nb, database.data());
```

3) Search the index
```
index.search(nq, queries.data(), k, dis.data(), nns.data());
```

For details on testing MIRAGE on the dataset of your choice, please look at other example files such as demos/demo_sift1M.cpp for reading data in the fvecs/ivecs format.