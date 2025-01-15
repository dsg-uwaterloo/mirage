/**
* Copyright (c) Facebook, Inc. and its affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexMIRAGE.h>

using namespace std::chrono;

int main(void) {
   // dimension of the vectors to index
   int d = 128;

   // size of the data we plan to index
   size_t nb = 1000000;

   std::mt19937 rng(12345);

   // make the index object
   faiss::IndexMirage index(d);
   index.mirage.S = 32;
   index.mirage.R = 4;
   index.mirage.iter = 15;
   index.verbose = true;


   // generate labels by IndexFlat
   faiss::IndexFlat bruteforce(d, faiss::METRIC_L2);

   std::vector<float> database(nb * d);
   for (size_t i = 0; i < nb * d; i++) {
       database[i] = rng() % 1024;
   }

   { // populating the index
       auto start_index = high_resolution_clock::now();
       index.add(nb, database.data());
       auto end_index = high_resolution_clock::now();
       auto indexing_time = duration_cast<milliseconds>(end_index - start_index).count();
       printf("Indexing completed in %ld ms\n", indexing_time);

       bruteforce.add(nb, database.data());
   }
   // number of queries
   size_t nq = 1000;

   { // searching the database
       printf("Searching ...\n");
       index.hierarchy.hnsw.efSearch = 1024;
       std::vector<float> queries(nq * d);
       for (size_t i = 0; i < nq * d; i++) {
           queries[i] = rng() % 1024;
       }

       int k = 100;
       std::vector<faiss::idx_t> nns(k * nq);
       std::vector<faiss::idx_t> gt_nns(k * nq);
       std::vector<float> dis(k * nq);

       auto start = high_resolution_clock::now();
       index.search(nq, queries.data(), k, dis.data(), nns.data());
       auto end = high_resolution_clock::now();

       // find exact kNNs by brute force search
       bruteforce.search(nq, queries.data(), k, dis.data(), gt_nns.data());

       int recalls = 0;
       for (size_t i = 0; i < nq; ++i) {
           for (int n = 0; n < k; n++) {
               for (int m = 0; m < k; m++) {
                   if (nns[i * k + n] == gt_nns[i * k + m]) {
                       recalls += 1;
                   }
               }
           }
       }
       float recall = 1.0f * recalls / (k * nq);
       auto t = duration_cast<microseconds>(end - start).count();
       int qps = nq * 1.0f * 1000 * 1000 / t;

       printf("Recall@%d: %f, QPS: %d\n", k, recall, qps);
   }

}
