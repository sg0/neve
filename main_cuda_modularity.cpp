// ***********************************************************************
//
//                              NEVE
//
// ***********************************************************************
//
//       Copyright (2019) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 



#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cfloat>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <cmath>

#ifdef LLNL_CALIPER_ENABLE
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#endif
#include "types.hpp"
#include "graph.hpp"
#include "graph_gpu.hpp"

#include <random>
#ifdef USE_32_BIT_GRAPH
typedef std::mt19937 Mt19937;
#else
typedef std::mt19937_64 Mt19937;
#endif

unsigned seed;

// A lot of print diagnostics is lifted from
// the STREAM benchmark.

static std::string inputFileName;
static GraphElem nvRGG = 0;
static int generateGraph = 0;

static GraphWeight randomEdgePercent = 0.0;
static bool randomNumberLCG = false;

// parse command line parameters
static void parseCommandLine(int argc, char** argv);

static void randomize_weights(GraphWeight* w, const GraphElem& ne)
{
    //std::random_device dev;
    Mt19937 rng;
    //std::default_random_engine rng;
    std::uniform_real_distribution<GraphWeight> distribution(0.,1.);

    for(GraphElem i = 0; i < ne; ++i)
        w[i] = distribution(rng);
}

static void set_random_commIds(GraphElem* commIds, const GraphElem& nv)
{
    //std::random_device dev;
    Mt19937 rng;
    //std::default_random_engine rng;
    std::uniform_int_distribution<GraphElem> distribution(0,nv/16+1);

    for(GraphElem i = 0; i < nv; ++i)
        //commIds[i] = distribution(rng);
        commIds[i] = i;
}
typedef struct EdgeKey
{
    GraphElem id;
    GraphElem e;
    GraphWeight w;
} EdgeKey;

bool compare_edge_key (EdgeKey a, EdgeKey b)
{
    return (a.id != b.id) ? (a.id < b.id) : (a.e < b.e);
}

static void sort_edges_by_commids(GraphWeight* weights, GraphElem* edges, GraphElem* indices, GraphElem* commIds, const GraphElem& nv)
{
    #pragma omp parallel for
    for(GraphElem v = 0; v < nv; ++v)
    {
        GraphElem start = indices[v];
        GraphElem end = indices[v+1];
        EdgeKey* array = new EdgeKey[end-start];
        for(GraphElem i = start; i < end; ++i)
            array[i-start] = {commIds[edges[i]], edges[i], weights[i]};
        std::stable_sort(array, array+end-start, compare_edge_key);
        for(GraphElem i = start; i < end; ++i)
        {
            edges[i] = array[i-start].e;
            weights[i] = array[i-start].w;
        }
        delete [] array; 
    }
}

int main(int argc, char **argv)
{
    parseCommandLine(argc, argv);
 
    Graph* g = nullptr;
    
    // generate graph only supports RGG as of now
    if (generateGraph) 
    { 
        GenerateRGG gr(nvRGG);
        g = gr.generate(randomNumberLCG, true /*isUnitEdgeWeight*/, randomEdgePercent);
    }
    else // read input graph 
    {   
        BinaryEdgeList rm;
        g = rm.read(inputFileName);
        std::cout << "Input file: " << inputFileName << std::endl;
    }

#if defined(PRINT_GRAPH_EDGES)        
    g->print();
#endif
    g->print_stats();
    assert(g != nullptr);

    const GraphElem nv = g->get_nv();
    const GraphElem ne = g->get_ne();

    g->nbrscan();
    g->nbrscan_edges();
    randomize_weights(g->get_edge_weights(), ne);

    GraphElem* edges = new GraphElem[ne];
    GraphWeight* weights = new GraphWeight[ne];

    GraphElem* edges_g = g->get_edges();
    GraphWeight* weights_g = g->get_edge_weights();
    GraphElem* indices = g->get_index_ranges();

    std::copy(edges_g, edges_g+ne, edges);
    std::copy(weights_g, weights_g+ne, weights);

    GraphGPU* g_gpu = new GraphGPU(g);

    g_gpu->sum_vertex_weights();

    GraphElem* commIds = new GraphElem[nv];
    set_random_commIds(commIds, nv);

    g_gpu->set_communtiy_ids(commIds);
    //g_gpu->singleton_partition();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    g_gpu->compute_modularity();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "sorting+modularity computation time on GPU is " << time*1E-03 << " s" << std::endl;

    /*t0 = omp_get_wtime();
    sort_edges_by_commids(edges, indices, commIds, nv);
    t1 = omp_get_wtime();
    std::cout << "sorting time on CPU is " << t1-t0 << " s" << std::endl;

    //err = 0;
    #pragma omp parallel for reduction(+:err)
    for(GraphElem i = 0; i < ne; ++i)
        err += std::pow(edges_g[i]-edges[i],2);
    std::cout << "error of the sorting edges: " << err << std::endl;*/
    //for(GraphElem i = indices[0]; i < indices[1]; ++i)
    //     std::cout << commIds[edges_g[i]] << " " << commIds[edges[i]] << std::endl;

    //for(GraphElem i = indices[0]; i < indices[1]; ++i)
     //   std::cout << weights_g[i] << " " << weights[i] << std::endl;


    delete [] commIds;
    delete [] edges;
    delete [] weights;

    delete g;
    delete g_gpu;

    return 0;
}

void parseCommandLine(int argc, char** const argv)
{
  int ret;
  optind = 1;
  bool help_text = false;

  if (argc == 1)
  {
      nvRGG = DEFAULT_NV;
      generateGraph = (nvRGG > 0)? true : false; 
  }
  else
  {
      while ((ret = getopt(argc, argv, "f:n:lp:h")) != -1) 
      {
          switch (ret) {
              case 'f':
                  inputFileName.assign(optarg);
                  break;
              case 'n':
                  nvRGG = atol(optarg);
                  if (nvRGG > 0)
                      generateGraph = true; 
                  break;
              case 'l':
                  randomNumberLCG = true;
                  break;
              case 'p':
                  randomEdgePercent = atof(optarg);
                  break;
              case 'h':
                  std::cout << "Set OMP_NUM_THREADS (max threads reported: " << omp_get_max_threads() << ") and affinity." << std::endl;
                  std::cout << "Usage [1] (use real-world file): ./neve_threads [-l] [-f /path/to/binary/file.bin] (see README)" << std::endl;
                  std::cout << "Usage [2] (use synthetic graph): ./neve_threads [-n <#vertices>] [-l] [-p <\% extra edges>]" << std::endl;
                  help_text = true;
                  break;
              default:
                  std::cout << "Please check the passed options." << std::endl;
                  break;
          }
      }
  }

  if (help_text)
      std::exit(EXIT_SUCCESS);

  if (!generateGraph && inputFileName.empty()) 
  {
      std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
      std::abort();
  }
   
  if (!generateGraph && randomNumberLCG) 
  {
      std::cerr << "Must specify -n <#vertices> for graph generation using LCG." << std::endl;
      std::abort();
  } 
   
  if (!generateGraph && (randomEdgePercent > 0.0)) 
  {
      std::cerr << "Must specify -n <#vertices> for graph generation first to add random edges to it." << std::endl;
      std::abort();
  } 
  
  if (generateGraph && ((randomEdgePercent < 0.0) || (randomEdgePercent >= 100.0))) 
  {
      std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
      std::abort();
  }
} // parseCommandLine
