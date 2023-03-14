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

#pragma once
#ifndef ALGOS_HPP
#define ALGOS_HPP
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <climits>
#include <array>
#include <unordered_map>

#include "utils.hpp"

// This file will contain algorithm implementations to be run on
// shared-memory or serial, using a compact CSR defined in 
// graph.hpp (which is essentially a subset of the Graph class)

// maximal edge graph matching
// https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8820975

class MaxEdgeMatching
{
    public:
        MaxEdgeMatching(CSR* g): 
            g_(g), edge_active_(0), 
            mate_(0), D_(0), M_(0) 
        {            
            nv_ = g_->get_nv();
            ne_ = g_->get_ne();
            
            edge_active_.resize(ne_, '1');
            mate_.resize(nv_, -1);           
        }

        ~MaxEdgeMatching() 
        {
            edge_active_.clear();
            D_.clear();
            M_.clear();
            mate_.clear();
        }
       
        MaxEdgeMatching(const MaxEdgeMatching &other) = delete;
        MaxEdgeMatching& operator=(const MaxEdgeMatching& d) = delete;
 
        char const& get_active_edge(GraphElem const index) const
        { return edge_active_[index]; }
         
        char& get_active_edge(GraphElem const index)
        { return edge_active_[index]; }
 
        // #edges in matched set  
        GraphElem get_mcount() const
        {
          GraphElem count = 0;
          for (GraphElem i = 0; i < M_.size(); i++)
            if ((M_[i].ij_[0] != -1) && (M_[i].ij_[1] != -1))
              count++;
          return count;
        }
        
        void print_M() const
        {
            std::cout << "Matched vertices: " << std::endl;
            for (GraphElem i = 0; i < M_.size(); i++)
            {
              if ((M_[i].ij_[0] != -1) && (M_[i].ij_[1] != -1))
              {
                std::cout << M_[i].ij_[0] << " ---- " << M_[i].ij_[1] << std::endl;
              }
            }
        }

        void flatten_M(std::vector<GraphElem>& matv) const
        {
            for (GraphElem i = 0; i < M_.size(); i++)
            {
              if ((M_[i].ij_[0] != -1) && (M_[i].ij_[1] != -1))
              {
                matv.push_back(M_[i].ij_[0]); 
                matv.push_back(M_[i].ij_[1]);
              }
            }
        }
         
        // if mate[mate[v]] == v then
        // we're good
        void check_results()
        {
            bool success = true;
            for (GraphElem i = 0; i < M_.size(); i++)
            {
              if ((M_[i].ij_[0] != -1) && (M_[i].ij_[1] != -1))
              {
                if ((mate_[mate_[M_[i].ij_[0]]] != M_[i].ij_[0])
                    || (mate_[mate_[M_[i].ij_[1]]] != M_[i].ij_[1]))
                {
                  std::cout << "\033[1;31mValidation FAILED.\033[0m" << std::endl; 
                  std::cout << "mate_[mate_[" << M_[i].ij_[0] << "]] != " << M_[i].ij_[0] << std::endl;
                  std::cout << "mate_[mate_[" << M_[i].ij_[1] << "]] != " << M_[i].ij_[1] << std::endl;
                  success = false;
                  break;
                }
              }
            }
            if (success)
                std::cout << "\033[1;32mValidation SUCCESS.\033[0m" << std::endl;
        }
        
        inline void heaviest_edge_unmatched(GraphElem v, Edge& max_edge, GraphElem x = -1)
        {
            GraphElem e0, e1;
            g_->edge_range(v, e0, e1);

            for (GraphElem e = e0; e < e1; e++)
            {
              Edge const& edge = g_->get_edge(e);
              char const& active = get_active_edge(e);
              if (active == '1')
              {
                if (edge.tail_ == x)
                  continue;

                if ((mate_[edge.tail_] == -1) 
                    || (mate_[mate_[edge.tail_]] 
                      != edge.tail_))
                {
                  if (edge.weight_ > max_edge.weight_)
                    max_edge = edge;

                  // break tie using vertex index
                  if (edge.weight_ == max_edge.weight_)
                  {
                    if (edge.tail_ > max_edge.tail_)
                      max_edge = edge;
                  }
                }
              }
            }
        }

        // check if mate[x] = v and mate[v] != x
        // if yes, compute mate[x]
        inline void update_mate(GraphElem v)
        {
            GraphElem e0, e1;
            g_->edge_range(v, e0, e1);
            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                GraphElem const& x = edge.tail_;

                // check if vertex is already matched
                auto result = std::find_if(M_.begin(), M_.end(), 
                        [&](EdgeTuple const& et) 
                        { return (((et.ij_[0] == v) || (et.ij_[1] == v)) && 
                                ((et.ij_[0] == x) || (et.ij_[1] == x))); });
                
                //  mate[x] == v and (v,x) not in M
                if ((mate_[x] == v) && (result == std::end(M_)))
                {
                    Edge x_max_edge;
                    heaviest_edge_unmatched(x, x_max_edge, v);
                    GraphElem y = mate_[x] = x_max_edge.tail_;

                    if (y == -1) // if x has no neighbor other than v
                        continue;

                    if (mate_[y] == x) // matched
                    {
                      D_.push_back(x);
                      D_.push_back(y);
                      M_.emplace_back(x, y, x_max_edge.weight_);

                      deactivate_edge(x, y);
                    }
                }
            }
        }

        // deactivate edge x -- y
        inline void deactivate_edge(GraphElem x, GraphElem y)
        {
            GraphElem e0, e1;
            g_->edge_range(x, e0, e1);
            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                char& active = get_active_edge(e);
                if (edge.tail_ == y)
                {
                    active = '0';
                    break;
                }
            }
        }

        // maximal edge matching
        void match()
        {
            // phase #1: compute max edge for every vertex
            for (GraphElem v = 0; v < nv_; v++)
            {
                Edge max_edge;
                heaviest_edge_unmatched(v, max_edge);

                GraphElem u = mate_[v] = max_edge.tail_; // v's mate

                if (u != -1)
                {  
                    // is mate[u] == v?
                    if (mate_[u] == v) // matched
                    {
                        D_.push_back(u);
                        D_.push_back(v);
                        M_.emplace_back(u, v, max_edge.weight_);

                        deactivate_edge(v, u);
                        deactivate_edge(u, v);
                    }
                }
            }

            // phase 2: update matching and match remaining vertices
            while(1)
            {     
                if (D_.size() == 0) 
                    break;
                GraphElem v = D_.back();
                D_.pop_back();
                update_mate(v);
            } 
        } 

    private:
        CSR* g_;
        GraphElem nv_, ne_;
        std::vector<char> edge_active_;
        std::vector<GraphElem> mate_, D_;
        std::vector<EdgeTuple> M_;  
};

#ifndef DEF_BFS_BUFSIZE
#define DEF_BFS_BUFSIZE (256)
#endif

#ifndef DEF_BFS_ROOTS
#define DEF_BFS_ROOTS (64)
#endif

#ifndef DEF_BFS_SEED
#define DEF_BFS_SEED (2)
#endif

class BFS
{
    public:
        BFS(Graph* g): 
            g_(g), visited_(nullptr), pred_(nullptr), bufsize_(DEF_BFS_BUFSIZE),
            comm_(MPI_COMM_NULL), rank_(MPI_PROC_NULL), size_(0), targets_(0), 
            pdegree_(0), pindex_(0), ract_(0), sact_(nullptr), sctr_(nullptr),
            sbuf_(nullptr), rbuf_(nullptr), sreq_(nullptr), rreq_(MPI_REQUEST_NULL), 
            oldq_(nullptr), newq_(nullptr), nranks_done_(0), rreq_act_(0),
            newq_count_(0), oldq_count_(0), seed_(DEF_BFS_SEED)
        {
            const GraphElem lnv = g_->get_lnv();
            comm_ = g_->comm();

            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);

                if ((e0 + 1) == e1)
                    continue;

                for (GraphElem m = e0; m < e1; m++)
                {
                    Edge const& edge_m = g_->get_edge(m);
                    const int owner = g_->get_owner(edge_m.tail_);

                    if (owner != rank_)
                    {
                        if (std::find(targets_.begin(), targets_.end(), owner) 
                                == targets_.end())
                            targets_.push_back(owner);
                    }
                }
            }

            MPI_Barrier(comm_);

            pdegree_ = targets_.size();

            for (int i = 0; i < pdegree_; i++)
                pindex_.insert({targets_[i], i});

            visited_ = new GraphElem[lnv];
            pred_    = new GraphElem[lnv];
            oldq_    = new GraphElem[lnv];
            newq_    = new GraphElem[lnv];
            rbuf_    = new GraphElem[bufsize_*2];
            sbuf_    = new GraphElem[pdegree_*bufsize_*2];
            sctr_    = new GraphElem[pdegree_];
            sreq_    = new MPI_Request[pdegree_];
            sact_    = new GraphElem[pdegree_];

            std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
            std::fill(sctr_, sctr_ + pdegree_, 0);
            std::fill(sact_, sact_ + pdegree_, 0);
            std::fill(oldq_, oldq_ + lnv, -1);
            std::fill(newq_, newq_ + lnv, -1);
            std::fill(pred_, pred_ + lnv, -1);
            std::fill(visited_, visited_ + lnv, 0);
        }

        ~BFS() 
        {
            delete []visited_;
            delete []pred_;
            delete []oldq_;
            delete []newq_;
            delete []rbuf_;
            delete []sbuf_;
            delete []sctr_;
            delete []sreq_;
            delete []sact_;
        }

        void set_visited(GraphElem v) { visited_[g_->global_to_local(v)] = 1; }
        GraphElem test_visited(GraphElem v) const { return visited_[g_->global_to_local(v)]; } 

        void process_msgs()
        {
            /* Check all MPI requests and handle any that have completed. */
            /* Test for incoming vertices to put onto the queue. */
            while (rreq_act_) 
            {
                int flag;
                MPI_Status st;
                MPI_Test(&rreq_, &flag, &st);
                if (flag) 
                {
                    ract_ = 0;
                    int count;
                    MPI_Get_count(&st, MPI_GRAPH_TYPE, &count);

                    /* count == 0 is a signal from a rank that it is done sending to me
                     * (using MPI's non-overtaking rules to keep that signal after all
                     * "real" messages. */
                    if (count == 0) 
                    {
                        ++nranks_done_;
                    } 
                    else 
                    {
                        for (int j = 0; j < count; j += 2) 
                        {
                            GraphElem tgt = rbuf_[j];
                            GraphElem src = rbuf_[j + 1];

                            /* Process one incoming edge. */
                            assert (g_->owner(tgt) == rank_);
                            if (!test_visited(tgt)) 
                            {
                                set_visited(tgt);
                                pred_[tgt] = src;
                                newq_[newq_count_++] = tgt;
                            }
                        }
                    }

                    /* Restart the receive if more messages will be coming. */
                    if (nranks_done_ < pdegree_) 
                    {
                        MPI_Irecv(rbuf_, bufsize_ * 2, MPI_GRAPH_TYPE, MPI_ANY_SOURCE, 0, comm_, &rreq_);
                        ract_ = 1;
                    }
                } 
                else 
                    break;
            }

            /* Mark any sends that completed as inactive so their buffers can be
             * reused. */
            for (int c = 0; c < pdegree_; ++c) 
            {
                if (sact_[c]) 
                {
                    int flag;
                    MPI_Test(&sreq_[c], &flag, MPI_STATUS_IGNORE);
                    if (flag) 
                        sact_[c] = 0;
                }
            }
        }

        void nbsend(GraphElem owner)
        {
            MPI_Isend(&sbuf_[pindex_[owner] * bufsize_ * 2], bufsize_ * 2, MPI_GRAPH_TYPE, 
                    owner, 0, comm_, &sreq_[pindex_[owner]]);

            sact_[pindex_[owner]] = 1;
            sctr_[pindex_[owner]] = 0;
        }

        void nbsend_zero(GraphElem owner)
        {
            MPI_Isend(nullptr, 0, MPI_DATATYPE_NULL, owner, 0, comm_, &sreq_[pindex_[owner]]);

            sact_[pindex_[owner]] = 1;
            sctr_[pindex_[owner]] = 0;
        }

        // reimplementation of graph500 BFS
        GraphElem run_bfs(GraphElem root) 
        {
            GraphElem edge_counts = 0;

#if defined(USE_ALLREDUCE_FOR_EXIT)
            GraphElem global_newq_count;
#else      
            bool done = false, nbar_active = false; 
            MPI_Request nbar_req = MPI_REQUEST_NULL;
#endif
            /* Mark the root and put it into the queue. */
            if (g_->owner(root) == rank_) 
            {
                set_visited(root);
                pred[g_->global_to_local(root)] = root;
                oldq_[oldq_count_++] = root;
            }

            process_msgs();

#if defined(USE_ALLREDUCE_FOR_EXIT)
                while(1)
#else
                while(!done)
#endif
                {
                    memset(sctr_, 0, pdegree_ * sizeof(GraphElem));
                    nranks_done = 1; /* I never send to myself, so I'm always done */

                    /* Start the initial receive. */
                    if (nranks_done < pdegree_) 
                    {
                        MPI_Irecv(rbuf_, bufsize_ * 2, MPI_GRAPH_TYPE, MPI_ANY_SOURCE, 0, comm_, &rreq_);
                        ract_ = 1;
                    }

                    /* Step through the current level's queue. */
                    for (GraphElem i = 0; i < oldq_count_; ++i) 
                    {
                        process_msgs();

                        assert (g_->owner(oldq_[i]) == rank_);
                        assert (pred[g_->global_to_local(oldq_[i])] >= 0 && pred[g_->global_to_local(oldq_[i])] < g_->lnv());
                        GraphElem src = oldq_[i];

                        /* Iterate through its incident edges. */
                        GraphElem e0, e1;
                        g_->edge_range(g_->global_to_local(src), e0, e1);

                        if ((e0 + 1) == e1)
                            continue;

                        for (GraphElem m = e0; m < e1; m++)
                        {
                            Edge const& edge = g_->get_edge(m);
                            const int owner = g_->get_owner(edge.tail_);
                            const GraphElem pidx = pindex_[owner];

                            if (owner == rank_)
                            {
                                if (!test_visited(edge.tail_)) 
                                {
                                    set_visited(edge.tail_);
                                    pred[g_->global_to_local(edge.tail_)] = src;
                                    newq_[newq_count_++] = g_->global_to_local(edge.tail_);
                                    edge_count++;
                                }
                            }
                            else
                            {
                                /* Wait for buffer to be available */
                                while (sact_[pidx]) 
                                    process_msgs();

                                GraphElem c = sctr_[pidx];
                                sbuf_[pidx * bufsize_ * 2 + c]     = edge.tail_;
                                sbuf_[pidx * bufsize_ * 2 + c + 1] = src;
                                sctr[pidx] += 2;

                                if (sctr[pidx] == (bufsize_ * 2))
                                    nbsend(owner);
                            }
                        }
                    }

                    /* Flush any coalescing buffers that still have messages. */
                    for (int const& p : targets_)
                    {
                        if (sctr_[pindex_[p]] != 0) 
                        {
                            while (sact_[pindex_[p]]) 
                                process_msgs();

                            nbsend(p);
                        }

                        /* Wait until all sends to this destination are done. */
                        while (sact_[pindex_[p]]) 
                            process_msgs();

                        /* Tell the destination that we are done sending to them. */
                        /* Signal no more sends */
                        nbsend_zero(p);

                        while (sact_[pindex_[p]]) 
                            process_msgs();
                    }

                    /* Wait until everyone else is done (and thus couldn't send us any more
                     * messages). */
                    while (nranks_done_ < pdegree_) 
                        process_msgs();

                    /* Test globally if all queues are empty. */
#if defined(USE_ALLREDUCE_FOR_EXIT)
                    MPI_Allreduce(&newq_count_, &global_newq_count, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);

                    /* Quit if they all are empty. */
                    if (global_newq_count == 0) 
                        break;
#else
                    if (nbar_active)
                    {
                        int test_nbar = -1;
                        MPI_Test(&nbar_req, &test_nbar, MPI_STATUS_IGNORE);
                        done = !test_nbar ? false : true;
                    }
                    else
                    {
                        if (newq_count_ == 0)
                        {
                            MPI_Ibarrier(comm_, &nbar_req);
                            nbar_active = true;
                        }
                    }
#endif

                    /* Swap old and new queues; clear new queue for next level. */
                    GraphElem *tmp = oldq_; 
                    oldq_ = newq_; 
                    newq_ = temp;

                    oldq_count_ = newq_count_;
                    newq_count_ = 0;
                }

                return edge_count;
        }

        void run_test(GraphElem nbfs_roots=DEF_BFS_ROOTS)
        {

            default_random_engine eng(seed_);
            uniform_int_distribution<GraphElem> uid(0, g_->nv()-1); 

            std::vector<GraphElem> bfs_roots;
            std::vector<double> bfs_times(nbfs_roots);

            bfs_roots.reserve(nbfs_roots);
            std::generate_n(std::back_inserter(bfs_roots), nbfs_roots, uid(eng));

            int test_ctr = 0;
            GraphElem ecg; /* Total edge visitations. */

            for (GraphElem const& r : bfs_roots)
            {
                if (rank_ == 0) 
                    fprintf(stderr, "Running BFS %d\n", test_ctr);

                /* Clear the pred array. */
                memset(pred_, 0, g_->lnv() * sizeof(GraphElem));

                /* Do the actual BFS. */
                double bfs_start = MPI_Wtime();
                GraphElem ec = run_bfs(r);
                double bfs_stop = MPI_Wtime();
                bfs_times[test_ctr] = bfs_stop - bfs_start;
                MPI_Allreduce(&ec, &ecg, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);

                if (rank == 0) 
                    fprintf(stderr, "Time, TEPS for BFS %d is %f, %g\n", test_ctr, bfs_times[test_ctr], (ecg / bfs_times[test_ctr]));
                test_ctr++;
            }
        }

    private:
        Graph* g_;

        int rank_, size_;
        std::vector<int> targets_;
        std::unordered_map<int, int> pindex_, mate_; 
        MPI_Comm comm_;

        GraphElem bufsize_, pdegree_, newq_count_, oldq_count_, nranks_done_, ract_, seed_;
        GraphElem *sbuf_, *rbuf_, *pred_, *visited_, *oldq_, *newq_, *sctr_, *sact_;
        MPI_Request *sreq_, rreq_;
};

#endif
