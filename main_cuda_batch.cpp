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
#include "graph_gpu.hpp"
unsigned seed;
#include "graph.hpp"
#include "cuda_wrapper.hpp"
#include <random>
#ifdef USE_32_BIT_GRAPH
typedef std::mt19937 Mt19937;
#else
typedef std::mt19937_64 Mt19937;
#endif

// A lot of print diagnostics is lifted from
// the STREAM benchmark.

static std::string inputFileName;
static GraphElem nvRGG = 0;
static int generateGraph = 0;

static GraphWeight randomEdgePercent = 0.0;
static bool randomNumberLCG = false;

// parse command line parameters
static void parseCommandLine(int argc, char** argv);

static void randomize_weights(Edge* w, const GraphElem& ne)
{
    //std::random_device dev;
    Mt19937 rng;
    //std::default_random_engine rng;
    std::uniform_real_distribution<GraphWeight> distribution(0.,1.);

    for(GraphElem i = 0; i < ne; ++i)
        w[i].weight_ = distribution(rng);
}

int main(int argc, char **argv)
{
    double t0, td, td0, td1;

    parseCommandLine(argc, argv);
 
    Graph* g = nullptr;
    
    td0 = omp_get_wtime();

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

    randomize_weights((Edge*)(g->get_edge_list()), g->get_num_edges());

    td1 = omp_get_wtime();
    td = td1 - td0;

    if (!generateGraph)
        std::cout << "Time to read input file and create graph (in s): " 
            << td << std::endl;
    else
        std::cout << "Time to generate graph of " 
            << nvRGG << " vertices (in s): " << td << std::endl;

#ifdef LLNL_CALIPER_ENABLE
    cali_config_set("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process");
#endif

    // nbrscan: 2*nv*(sizeof GraphElem) + 2*ne*(sizeof GraphWeight) + (2*ne*(sizeof GraphElem + GraphWeight)) 
    // nbrsum : 2*nv*(sizeof GraphElem) + 3*ne*(sizeof GraphWeight) + (2*ne*(sizeof GraphElem + GraphWeight)) 
    // nbrmax : 2*nv*(sizeof GraphElem) + 2*ne*(sizeof GraphWeight) + nv*(sizeof GraphWeight) + (2*ne*(sizeof GraphElem + GraphWeight)) 
    const GraphElem nv = g->get_nv();
    const GraphElem ne = g->get_ne();

#ifdef EDGE_AS_VERTEX_PAIR
    const std::size_t count_nbrscan = ne*((2*sizeof(GraphElem) + sizeof(GraphWeight)) + 2*sizeof(GraphWeight)); 
    const std::size_t count_nbrsum = ne*((2*sizeof(GraphElem) + sizeof(GraphWeight)) + 3*sizeof(GraphElem)); 
    const std::size_t count_nbrmax = ne*((2*sizeof(GraphElem) + sizeof(GraphWeight)) + 2*sizeof(GraphWeight)); 
#else
    const std::size_t count_nbrscan = ne*((sizeof(GraphElem) + sizeof(GraphWeight)) + 2*sizeof(GraphWeight)); 
    const std::size_t count_nbrsum = ne*((sizeof(GraphElem) + sizeof(GraphWeight)) + 3*sizeof(GraphElem)); 
    const std::size_t count_nbrmax = ne*((sizeof(GraphElem) + sizeof(GraphWeight)) + 2*sizeof(GraphWeight)); 
#endif

    std::printf("Total memory required (Neighbor Scan) = %.1f KiB = %.1f MiB = %.1f GiB.\n",
        ( (double) (count_nbrscan) / 1024.0),
        ( (double) (count_nbrscan) / 1024.0/1024.0),
        ( (double) (count_nbrscan) / 1024.0/1024.0/1024.0));
    std::printf("Total memory required (Neighbor Sum ) = %.1f KiB = %.1f MiB = %.1f GiB.\n",
        ( (double) (count_nbrsum) / 1024.0),
        ( (double) (count_nbrsum) / 1024.0/1024.0),
        ( (double) (count_nbrsum) / 1024.0/1024.0/1024.0));
    std::printf("Total memory required (Neighbor Max ) = %.1f KiB = %.1f MiB = %.1f GiB.\n",
        ( (double) (count_nbrmax) / 1024.0),
        ( (double) (count_nbrmax) / 1024.0/1024.0),
        ( (double) (count_nbrmax) / 1024.0/1024.0/1024.0));

#ifdef LLNL_CALIPER_ENABLE
#else 
    std::printf("Each kernel will be executed %d times.\n", NTIMES);
    std::printf(" The *best* time for each kernel (excluding the first iteration)\n");
    std::printf(" will be used to compute the reported bandwidth.\n");
#endif

    int quantum;
    if  ( (quantum = omp_get_wtick()) >= 1)
        std::printf("Your clock granularity/precision appears to be "
                "%d microseconds.\n", quantum);
    else 
    {
        std::printf("Your clock granularity appears to be "
                "less than one microsecond.\n");
        quantum = 1;
    }

    t0 = omp_get_wtime();
    g->nbrscan();
    t0 = 1.0E6 * (omp_get_wtime() - t0);
    std::printf("Each test below will take on the order"
        " of %d microseconds.\n", (int) t0);
    std::printf("   (= %d clock ticks)\n", (int) (t0/quantum) );
    std::printf("Increase the size of the graph if this shows that\n");
    std::printf("you are not getting at least 20 clock ticks per test.\n");

#ifdef LLNL_CALIPER_ENABLE
        g->nbrscan();
        g->nbrsum();
        g->nbrmax();
#else
    double times[3][NTIMES]; 
    double avgtime[3] = {0}, maxtime[3] = {0}, mintime[3] = {FLT_MAX,FLT_MAX,FLT_MAX};

    for (int k = 0; k < NTIMES; k++)
    {
        times[0][k] = omp_get_wtime();
        g->nbrscan();
        times[0][k] = omp_get_wtime() - times[0][k];
        times[1][k] = omp_get_wtime();
        g->nbrmax();
        times[1][k] = omp_get_wtime() - times[1][k];
        times[2][k] = omp_get_wtime();
        g->nbrsum();
        times[2][k] = omp_get_wtime() - times[2][k];
    }

    for (int k = 1; k < NTIMES; k++) // note -- skip first iteration
    {
        for (int j = 0; j < 3; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = std::min(mintime[j], times[j][k]);
            maxtime[j] = std::max(maxtime[j], times[j][k]);
        }
    }

    std::string label[3] = {"Neighbor Copy:    ", "Neighbor Max :    ", "Neighbor Add :    "};
    double bytes[3] = { (double)ne, (double)ne, (double)ne };

    printf("Function            Best Rate TEPs  Avg time     Min time     Max time\n");
    for (int j = 0; j < 3; j++) 
    {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);
        std::printf("%s%12.1f  %12.6f  %11.6f  %11.6f\n", label[j].c_str(),
                1.0E-06 * bytes[j]/mintime[j], avgtime[j], mintime[j],
                maxtime[j]);
    }
#endif

    //perform the batching gpu part
    float times_cuda[3][NTIMES];
    float copy_times[3][NTIMES]; 
    double avgtime_cuda[3] = {0}, maxtime_cuda[3] = {0}, mintime_cuda[3] = {FLT_MAX,FLT_MAX,FLT_MAX};

    GraphGPU* g_cuda = new GraphGPU(g);
    //for gpu timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    #ifdef CHECK
    GraphWeight* e_ws = new GraphWeight [ne];
    GraphWeight* v_ws = new GraphWeight [nv];

    g->nbrscan();

    GraphWeight* weights = g->get_edge_weights();
    std::copy(weights, weights+ne,e_ws);
    #endif
    for (int k = 0; k < NTIMES; k++)
    {
        cudaEventRecord(start, 0);
        float t = g_cuda->scan_edge_weights();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_cuda[0][k], start, stop);
        copy_times[0][k] = t;
    }
    //check results;
    #ifdef CHECK 
    GraphWeight err = 0.;
    for(GraphElem i = 0; i < ne; ++i)
        err += std::pow(weights[i]-e_ws[i],2);
    std::cout << "function scan error: " << err << std::endl;

    g->nbrmax();

    weights = g->get_vertex_weights();
    GraphWeight* weights_dev = (GraphWeight*)(g_cuda->get_vertex_weights());
    #endif
    //std::copy(weights, weights+nv, v_ws);
    for (int k = 0; k < NTIMES; k++)
    {
        cudaEventRecord(start, 0);
        float t = g_cuda->max_vertex_weights();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_cuda[1][k], start, stop);
        copy_times[1][k] = t;
    }
    //check resutls
    #ifdef CHECK
    CudaMemcpyDtoH(v_ws, weights_dev, sizeof(GraphWeight)*nv);
    err = 0.;
    for(GraphElem i = 0; i < nv; ++i)
        err += std::pow(weights[i]-v_ws[i],2);
    std::cout << "function of max weights: "<< err << std::endl;

    g->nbrsum();

    std::copy(weights, weights+nv, v_ws);
    #endif
    for (int k = 0; k < NTIMES; k++)   
    {   
        cudaEventRecord(start, 0);      
        float t = g_cuda->sum_vertex_weights();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_cuda[2][k], start, stop);
        copy_times[2][k] = t;
    }
    //check results
    #ifdef CHECK
    CudaMemcpyDtoH(v_ws, weights_dev, sizeof(GraphWeight)*nv);
    err = 0.;
    for(GraphElem i = 0; i < nv; ++i)
        err += std::pow(weights[i]-v_ws[i],2);
    std::cout << "function of sum weights: "<< err << std::endl;

    delete [] e_ws;
    delete [] v_ws;
    #endif

    for (int k = 1; k < NTIMES; k++) // note -- skip first iteration
    {
        for (int j = 0; j < 3; j++)
        {
            avgtime_cuda[j] = avgtime_cuda[j] + times_cuda[j][k];
            mintime_cuda[j] = std::min(mintime_cuda[j], (double)times_cuda[j][k]);
            maxtime_cuda[j] = std::max(maxtime_cuda[j], (double)times_cuda[j][k]);
        }
    }

    double avgtime_copy[3] = {0};
    double mintime_copy[3] = {FLT_MAX,FLT_MAX,FLT_MAX};
    double maxtime_copy[3] = {0};

    for (int k = 1; k < NTIMES; k++) // note -- skip first iteration
    {
        for (int j = 0; j < 3; j++)
        {
            avgtime_copy[j] = avgtime_copy[j] + copy_times[j][k];
            mintime_copy[j] = std::min(mintime_copy[j], (double)copy_times[j][k]);
            maxtime_copy[j] = std::max(maxtime_copy[j], (double)copy_times[j][k]);
        }
    }

    //std::string label[3] = {"Neighbor Copy:    ", "Neighbor Add :    ", "Neighbor Max :    "};
    //double bytes[3] = { (double)count_nbrscan, (double)count_nbrsum, (double)count_nbrmax };

    printf("				GPU Profile				  \n");
    printf("Function            Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int j = 0; j < 3; j++)
    {
        avgtime_cuda[j] = avgtime_cuda[j]/(double)(NTIMES-1);
        std::printf("%s%12.1f  %12.6f  %11.6f  %11.6f\n", label[j].c_str(),
                1.0E-06 * bytes[j]/mintime_cuda[j]*1.0E03, avgtime_cuda[j]*1.0E-03, 
                mintime_cuda[j]*1.0E-03,maxtime_cuda[j]*1.0E-03);
    }

    double copy_bytes[3] = {(double)sizeof(Edge)*ne, (double)sizeof(GraphWeight)*ne, (double)sizeof(GraphWeight)*ne};
    printf("                            GPU Copy Profile                           \n");
    printf("Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int j = 0; j < 3; j++)
    {
        avgtime_copy[j] = avgtime_copy[j]/(double)(NTIMES-1);
        std::printf("%11.6f %11.6f  %11.6f  %11.6f\n",
                1.0E-06 * copy_bytes[j]/mintime_copy[j]*1.0E03, avgtime_copy[j]*1.0E-03,
                mintime_copy[j]*1.0E-03,maxtime_copy[j]*1.0E-03);
    }

    delete g_cuda;
    delete g;
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
