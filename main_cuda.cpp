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
unsigned seed;
#include "types.hpp"
#include "graph.hpp"

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
    //perform the gpu part
    float times_cuda[3][NTIMES]; 
    double avgtime_cuda[3] = {0}, maxtime_cuda[3] = {0}, mintime_cuda[3] = {FLT_MAX,FLT_MAX,FLT_MAX};
    float copy_time = g->map_data_on_device();
#if defined(INCLUDE_TRANSFER_TIME)
#else
    copy_time = 0.0;
#endif
    //for gpu timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int k = 0; k < NTIMES; k++)
    {
        cudaEventRecord(start, 0);
        g->nbrscan_cuda();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_cuda[0][k], start, stop);

        cudaEventRecord(start, 0);
        g->nbrmax_cuda();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_cuda[1][k], start, stop);
        

        cudaEventRecord(start, 0);      
        g->nbrsum_cuda();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_cuda[2][k], start, stop);
    }

    for (int k = 1; k < NTIMES; k++) // note -- skip first iteration
    {
        for (int j = 0; j < 3; j++)
        {
            avgtime_cuda[j] = avgtime_cuda[j] + times_cuda[j][k];
            mintime_cuda[j] = std::min(mintime_cuda[j], (double)times_cuda[j][k]);
            maxtime_cuda[j] = std::max(maxtime_cuda[j], (double)times_cuda[j][k]);
        }
    }

    //std::string label[3] = {"Neighbor Copy:    ", "Neighbor Add :    ", "Neighbor Max :    "};
    //double bytes[3] = { (double)count_nbrscan, (double)count_nbrsum, (double)count_nbrmax };
#if defined(INCLUDE_TRANSFER_TIME)
    printf("                            GPU Copy Profile                          \n");
    float copy_size = (sizeof(Edge)*ne+sizeof(GraphElem)*(nv+1))/(1024.f*1024.f*1024.f);
    std::printf("Ave. Time: %12.6fs. Bandwidth %12.6fGB/s\n", copy_time*1.0E-03, copy_size/copy_time*1.0E03);
#else
#endif
    printf("				GPU Profile 				  \n");
    printf("Function            Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int j = 0; j < 3; j++)
    {
        avgtime_cuda[j] = avgtime_cuda[j]/(double)(NTIMES-1);
        std::printf("%s%12.1f  %12.6f  %11.6f  %11.6f\n", label[j].c_str(),
                1.0E-06 * bytes[j]/(copy_time+mintime_cuda[j])*1.0E03, (copy_time+avgtime_cuda[j])*1.0E-03, 
                (copy_time+mintime_cuda[j])*1.0E-03,(copy_time+maxtime_cuda[j])*1.0E-03);
    }
    //check whether the answer is correct
    g->check_results(); 
#endif
    //start testing the re-order edges
    g->map_data_release_device();
/*
    g->allocate(); 
 
    GraphElem* index_orders = new GraphElem[ne];  
    GraphElem* commIds = new GraphElem[nv];
    set_random_commIds(commIds, nv);
    randomize_weights(g->edge_weights_, ne);
    g->nbrscan_edges();
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    g->nbrsort_edges_by_commids(index_orders, commIds);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float sort_time;    
    cudaEventElapsedTime(&sort_time, start, stop);
    std::cout << "The Time of sorting is " << sort_time*1E-03 << " s" << std::endl;
    g->deallocate();

    delete [] index_orders;
    delete [] commIds;*/
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
