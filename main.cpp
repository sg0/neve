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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "comm.hpp"

static std::string inputFileName;
static std::string outputFileName;
static bool graphIsProcessGraph = false;
static bool writeOutputFile = false;
static int me, nprocs;
static int ranksPerNode = 1;
static GraphElem nvRGG = 0;
static int generateGraph = 0;

static GraphWeight randomEdgePercent = 0.0;
static long minSizeExchange = 0;
static long maxSizeExchange = 0;
static long maxNumGhosts = 0;
static bool readBalanced = false;
static bool hardSkip = false;
static bool randomNumberLCG = false;
static bool fallAsleep = false;

static int bwOption = 0, performWork = 0;
static int lttOption = -1; // default: no latency test
static bool performBWTest = false;
static bool performBWTestRMA = false;
static bool performLTTest = false;
static bool performWorkMax = false;
static bool performWorkSum = false;
static bool createRankOrder = false;
static int rankOrderType = 0;

static bool chooseSingleNbr = false;
static int processNbr = 0;
static bool shrinkGraph = false;
static float graphShrinkPercent = 0;

// parse command line parameters
static void parseCommandLine(int argc, char** argv);

int main(int argc, char **argv)
{
    double t0, t1, td, td0, td1;

    shmem_init();
    
    int mpi_init;
    MPI_Initialized(&mpi_init);
    if (!mpi_init) {
        MPI_Init(&argc, &argv);
    }
#if defined(SCOREP_USER_ENABLE)
    SCOREP_RECORDING_OFF();
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    // command line options
    MPI_Barrier(MPI_COMM_WORLD);
    parseCommandLine(argc, argv);
 
    Graph* g = nullptr;
    
    td0 = MPI_Wtime();

    // generate graph only supports RGG as of now
    if (generateGraph) 
    { 
        GenerateRGG gr(nvRGG);
        g = gr.generate(randomNumberLCG, true /*isUnitEdgeWeight*/, randomEdgePercent);
    }
    else 
    {   // read input graph
#ifndef SSTMAC
        BinaryEdgeList rm;
        if (graphIsProcessGraph) {
            if (me == 0) {
                std::cout << std::endl;
                std::cout << "Standard edge distribution (#vertices/p) while reading: " << std::endl;
                std::cout << inputFileName << std::endl;
            }
            g = rm.read(me, nprocs, ranksPerNode, inputFileName);
        } else {
            if (readBalanced == true)
            {
                if (me == 0)
                {
                    std::cout << std::endl;
                    std::cout << "Trying to balance the edge distribution (#edges/p) while reading: " << std::endl;
                    std::cout << inputFileName << std::endl;
                }
                g = rm.read_balanced(me, nprocs, ranksPerNode, inputFileName);
            }
            else
            {
                if (me == 0)
                {
                    std::cout << std::endl;
                    std::cout << "Standard edge distribution (#vertices/p) while reading: " << std::endl;
                    std::cout << inputFileName << std::endl;
                }
                g = rm.read(me, nprocs, ranksPerNode, inputFileName);
            }
        }
#else
#warning "SSTMAC is defined: Trying to load external graph binaries will FAIL."
#endif
    }

#if defined(PRINT_GRAPH_EDGES)        
    g->print();
#endif
    g->print_dist_stats();
    assert(g != nullptr);

    if (createRankOrder) 
    {
        t0 = MPI_Wtime();
        if (rankOrderType == 1)
            g->rank_order();
        else if (rankOrderType == 2)
            g->weighted_rank_order();
        else
            g->matching_rank_order(outputFileName, writeOutputFile);
        t1 = MPI_Wtime() - t0;
        double tr = 0.0;
        MPI_Reduce(&t1, &tr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (me == 0) 
            std::cout << "Time to create rank order file (in s): " << tr << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    td1 = MPI_Wtime();
    td = td1 - td0;

    double tdt = 0.0;
    MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (me == 0)  
    {
        if (!generateGraph)
            std::cout << "Time to read input file and create distributed graph (in s): " 
                << tdt << std::endl;
        else
            std::cout << "Time to generate distributed graph of " 
                << nvRGG << " vertices (in s): " << tdt << std::endl;
    }
    
    if (performBWTest || performBWTestRMA || lttOption != -1)
    {
        // Comm object can be instantiated
        // with iteration ranges and other 
        // info, see class Comm in comm.hpp
        if (maxSizeExchange == 0)
            maxSizeExchange = MAX_SIZE;
        if (minSizeExchange == 0)
            minSizeExchange = MIN_SIZE;

        Comm c(g, minSizeExchange, maxSizeExchange, graphShrinkPercent);

        MPI_Barrier(MPI_COMM_WORLD);

        t0 = MPI_Wtime();

        // ---------------------------------------
        // bandwidth tests
        // ---------------------------------------
        
        if (performBWTest) 
        {
            if (chooseSingleNbr)
            {
                if (me == 0)
                {
                    std::cout << "Choosing the neighborhood of process #" << processNbr 
                        << " for bandwidth test." << std::endl;
                }
                if (maxNumGhosts > 0)
                    c.p2p_bw_snbr(processNbr, maxNumGhosts);
                else
                    c.p2p_bw_snbr(processNbr);
            }
            else
            {
                if (hardSkip)
                    c.p2p_bw_hardskip(0);
                else
                    c.p2p_bw(0);
            }
        } else if (performBWTestRMA) 
        {
            if (hardSkip)
                c.p2p_bw_hardskip(1);
            else
                c.p2p_bw(1);
        }
        
        // done with bandwidth tests
        // ---------------------------------------
        
        
        
        // ---------------------------------------
        // latency tests
        // ---------------------------------------
        
        if (chooseSingleNbr) {
            // don't do the standard send/recv test later
            lttOption = -1;
            if (me == 0)
            {
                std::cout << "Choosing the neighborhood of process #" << processNbr 
                    << " for latency test." << std::endl;
            }
            c.p2p_lt_snbr(processNbr);
        }
        
        if (lttOption == 1 || lttOption == 2) {
#if SSTMAC
            lttOption = -1;
#warning "SSTMAC is defined: MPI3 neighborhood collectives are turned OFF."
#endif
        }
        
        switch (lttOption) {
        case 0: // Nonblocking send/recv
            if (fallAsleep) {
                if (me == 0)
                    std::cout << "Invoking (u)sleep for an epoch equal to #locally-owned-vertices" << std::endl;
                c.p2p_lt_usleep();
            }
            else if (performWorkSum) {
                if (me == 0)
                    std::cout << "Invoking work performing degree sum for #locally-owned-vertices" << std::endl;
                c.p2p_lt_worksum();
            }
            else if (performWorkMax) {
                if (me == 0)
                    std::cout << "Invoking work performing degree max for #locally-owned-vertices" << std::endl;
                c.p2p_lt_workmax();
            }
            else
                // run p2p_lt using nonblocking send/recv kernel
                c.p2p_lt(0);
            break;
            
        case 1: // MPI_Neighbor_alltoall
            c.nbr_ala_lt();
            break;
        case 2: // MPI_Neighbor_allgather
            c.nbr_aga_lt();
            break;
        case 3: // MPI RMA with MPI_Rput using flush
            c.p2p_lt(3);
            break;
        case 4: // MPI RMA with MPI_Rput using fence
            c.p2p_lt(4);
            break;
        case 5: // MPI with nonblocking consensus
            c.p2p_lt(7);
            break;
        case 6: // SHMEM with barrier
            c.p2p_lt(9);
            break;
        case 7: // SHMEM with put_signal
            c.p2p_lt(8);
            break;
        default:
            break;
        }
        
        // done with latency tests
        // ---------------------------------------

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        double p_tot = t1 - t0, t_tot = 0.0;

        MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, 
                MPI_SUM, 0, MPI_COMM_WORLD);
        if (me == 0) 
        {
            std::cout << "Average execution time (in s) for running the test on " << nprocs << " processes: " 
                << (double)(t_tot/(double)nprocs) << std::endl;
#ifndef SSTMAC
            std::cout << "Resolution of MPI_Wtime: " << MPI_Wtick() << std::endl;
#endif
        }

        c.destroy_nbr_comm();
    } // end latency/bandwidth tests

    MPI_Barrier(MPI_COMM_WORLD);
   
    shmem_finalize();
    
    int mpi_final;
    MPI_Finalized(&mpi_init);
    if (!mpi_final) {
         MPI_Finalize();
    }

    return 0;
}

void parseCommandLine(int argc, char** const argv)
{
  int ret;
  optind = 1;

  while ((ret = getopt(argc, argv, "f:y:r:n:lhp:m:x:bg:t:w:s:z:ud:o:a:")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
      break;
    case 'y':
      graphIsProcessGraph = true;
      break;
    case 'b':
      readBalanced = true;
      break;
    case 'r':
      ranksPerNode = atoi(optarg);
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
    case 'x':
      maxSizeExchange = atol(optarg);
      break;
    case 'm':
      minSizeExchange = atol(optarg);
      break;
    case 'g':
      maxNumGhosts = atol(optarg);
      break;
    case 'w':
      bwOption = atoi(optarg);
      if (bwOption == 0)
          // use nonblocking Send/Recv
          performBWTest = true;
      else if (bwOption == 1)
          // use MPI_Neighbor_alltoall
          performBWTestRMA = true;
      else
          performBWTest = true;
      break;
    case 't':
      lttOption = atoi(optarg);
      break;
    case 'd':
      performWork = atoi(optarg);
      if (performWork > 0)
          performWorkSum = true;
      else
          performWorkMax = true;
      break;
    case 'h':
      hardSkip = true;
      break;
    case 's':
      chooseSingleNbr = true;
      processNbr = atoi(optarg);
      break;
    case 'z':
      shrinkGraph = true;
      graphShrinkPercent = atof(optarg);
      break;
    case 'u':
      fallAsleep = true;
      break;
    case 'o':
      rankOrderType = atoi(optarg);
      createRankOrder = true;
      break;
    case 'a':
      outputFileName.assign(optarg);
      writeOutputFile = true;
      createRankOrder = true;
      break;
    default:
      assert(0 && "Should not reach here!!");
      break;
    }
  }
std::cout << "LTTOPTIOBN" << lttOption << std::endl;
  // warnings/info
  if (me == 0 && lttOption != -1 && maxNumGhosts) 
  {
      std::cout << "Setting the number of ghost vertices (-g <...>) has no effect for latency test."
          << std::endl;
  }
   
  if (me == 0 && generateGraph && readBalanced) 
  {
      std::cout << "Balanced read (option -b) is only applicable for real-world graphs. "
          << "This option does nothing for generated (synthetic) graphs." << std::endl;
  } 
   
  if (me == 0 && generateGraph && shrinkGraph && graphShrinkPercent > 0.0) 
  {
      std::cout << "Graph shrinking (option -z) is only applicable for real-world graphs. "
          << "This option does nothing for generated (synthetic) graphs." << std::endl;
  } 
   
  if (me == 0 && shrinkGraph && graphShrinkPercent <= 0.0) 
  {
      std::cout << "Graph shrinking (option -z) must be greater than 0.0. " << std::endl;
  }

  if (me == 0 && shrinkGraph && lttOption != -1)
  {
	  std::cout << "Graph shrinking is ONLY valid for bandwidth test, NOT latency test which just performs message exchanges across the process neighborhood of a graph." << std::endl;	  
  }

  if (me == 0 && lttOption != -1 && hardSkip)
  {
      std::cout << "The hard skip option to disable warmup and extra communication loops only affects the bandwidth test." << std::endl;
  }

  if (me == 0 && (!performLTTest) && (performWorkSum || performWorkMax))
  {
      std::cout << "Passing -d <> has no effect unless -t <> is passed as well. In other words, work-option is enabled only for latency tests." << std::endl;
  }
  
  if (me == 0 && chooseSingleNbr && lttOption != 0)
  {
      std::cout << "At present, only MPI Isend/Irecv communication is supported when a single process's neighborhood is selected." << std::endl;
  }
  
  if (me == 0 && lttOption > 7)
  {
      std::cout << "Valid values for latency test arguments are 0 (Isend/Irecv, the default case), 1 (Neighbor All-to-All), 2 (Neighbor All-Gather), and 3-5 (RMA)." << std::endl;
  }
  
  if (me == 0 && graphIsProcessGraph && readBalanced)
  {
      std::cout << "Can't do a balanced read with a process graph. Ignoring -b." << std::endl;
      readBalanced = false;
  }

  // errors
  if (me == 0 && (argc == 1)) 
  {
      std::cerr << "Must specify some options." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && !generateGraph && inputFileName.empty()) 
  {
      std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
   
  if (me == 0 && !generateGraph && randomNumberLCG) 
  {
      std::cerr << "Must specify -n for graph generation using LCG." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
   
  if (me == 0 && !generateGraph && (randomEdgePercent > 0.0)) 
  {
      std::cerr << "Must specify -n for graph generation first to add random edges to it." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
  
  if (me == 0 && generateGraph && ((randomEdgePercent < 0.0) || (randomEdgePercent >= 100.0))) 
  {
      std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }

  if (me == 0 && !chooseSingleNbr && (maxNumGhosts > 0)) 
  {
      std::cerr << "Fixing ghosts only allowed when a single neighborhood (-s <...>) is chosen." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && !generateGraph && shrinkGraph && (graphShrinkPercent != 0.0 && (graphShrinkPercent < 0.0 || graphShrinkPercent > 100.0))) 
  {
      std::cerr << "Allowable value of graph shrink percentage is 0.0...-100%." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
} // parseCommandLine
