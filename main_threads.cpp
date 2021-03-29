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

#ifdef LLNL_CALIPER_ENABLE
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#endif

#include "graph.hpp"

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
    double t0, t1, td, td0, td1;

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
    const GraphElem nv = g->get_nv();
    const GraphElem ne = g->get_ne();
#ifdef EDGE_AS_VERTEX_PAIR
    const std::size_t count_nbrscan = 2*nv*sizeof(GraphElem) + 2*ne*sizeof(GraphWeight) 
        + 2*ne*(sizeof(GraphElem) + sizeof(GraphElem) + sizeof(GraphWeight)); 
    const std::size_t count_nbrsum = 2*nv*sizeof(GraphElem) + 3*ne*sizeof(GraphWeight) 
        + 2*ne*(sizeof(GraphElem) + sizeof(GraphElem) + sizeof(GraphWeight));
#else
    const std::size_t count_nbrscan = 2*nv*sizeof(GraphElem) + 2*ne*sizeof(GraphWeight) 
        + 2*ne*(sizeof(GraphElem) + sizeof(GraphWeight)); 
    const std::size_t count_nbrsum = 2*nv*sizeof(GraphElem) + 3*ne*sizeof(GraphWeight) 
        + 2*ne*(sizeof(GraphElem) + sizeof(GraphWeight));
#endif

    std::printf("Total memory required (Neighbor Scan) = %.1f KiB = %.1f MiB = %.1f GiB.\n",
        ( (double) (count_nbrscan) / 1024.0),
        ( (double) (count_nbrscan) / 1024.0/1024.0),
        ( (double) (count_nbrscan) / 1024.0/1024.0/1024.0));
    std::printf("Total memory required (Neighbor Sum ) = %.1f KiB = %.1f MiB = %.1f GiB.\n",
        ( (double) (count_nbrsum) / 1024.0),
        ( (double) (count_nbrsum) / 1024.0/1024.0),
        ( (double) (count_nbrsum) / 1024.0/1024.0/1024.0));

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
#else
    double times[2][NTIMES]; 
    double avgtime[2] = {0}, maxtime[2] = {0}, mintime[2] = {FLT_MAX,FLT_MAX};

    for (int k = 0; k < NTIMES; k++)
    {
        times[0][k] = omp_get_wtime();
        g->nbrscan();
        times[0][k] = omp_get_wtime() - times[0][k];
        times[1][k] = omp_get_wtime();
        g->nbrsum();
        times[1][k] = omp_get_wtime() - times[1][k];
    }

    for (int k = 1; k < NTIMES; k++) // note -- skip first iteration
    {
        for (int j = 0; j < 2; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = std::min(mintime[j], times[j][k]);
            maxtime[j] = std::max(maxtime[j], times[j][k]);
        }
    }

    std::string label[2] = {"Neighbor Copy:    ", "Neighbor Add :    "};
    double bytes[2] = { (double)count_nbrscan, (double)count_nbrsum };

    printf("Function            Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int j = 0; j < 2; j++) 
    {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);
        std::printf("%s%12.1f  %12.6f  %11.6f  %11.6f\n", label[j].c_str(),
                1.0E-06 * bytes[j]/mintime[j], avgtime[j], mintime[j],
                maxtime[j]);
    }
#endif
    
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
