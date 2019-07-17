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
#ifndef COMM_HPP
#define COMM_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>

#define MAX_SIZE                (1<<22)
#define MIN_SIZE                (0)
#define LARGE_SIZE              8192
#define ZCI                     (1.96)

#define BW_LOOP_COUNT           100
#define BW_SKIP_COUNT           10
#define BW_LOOP_COUNT_LARGE     20
#define BW_SKIP_COUNT_LARGE     2

#define LT_LOOP_COUNT           10000
#define LT_SKIP_COUNT           100
#define LT_LOOP_COUNT_LARGE     1000
#define LT_SKIP_COUNT_LARGE     10

// TODO FIXME add a version that only considers x% of ghosts
// per process

class Comm
{
    public:

#define COMM_COMMON() \
        do { \
            comm_ = g_->get_comm(); \
            MPI_Comm_size(comm_, &size_); \
            MPI_Comm_rank(comm_, &rank_); \
            /* track ghosts not owned by me */ \
            const GraphElem lne = g_->get_lne(); \
            lnv_ = g_->get_lnv(); \
            int pdx = 0; \
            for (GraphElem i = 0; i < lnv_; i++) \
            { \
                GraphElem e0, e1; \
                g_->edge_range(i, e0, e1); \
                for (GraphElem e = e0; e < e1; e++) \
                { \
                    Edge const& edge = g_->get_edge(e); \
                    const int owner = g_->get_owner(edge.tail_); \
                    if (owner != rank_) \
                    { \
                        nghosts_ += 1; \
                        if (std::find(targets_.begin(), targets_.end(), owner) == targets_.end()) \
                        { \
                            targets_.push_back(owner); \
                            pindex_.insert({owner, pdx}); \
                            pdx += 1; \
                            nghosts_in_target_.resize(pdx, 0); \
                        } \
                        nghosts_in_target_[pindex_[owner]] += 1; \
                    } \
                } \
            } \
            degree_ = targets_.size(); \
            if (shrinkp_ > 0) \
            { \
                GraphElem new_nghosts = 0; \
                for (int p = 0; p < degree_; p++) \
                { \
                    nghosts_in_target_[p] = (shrinkp_ * nghosts_in_target_[p]) / 100; \
                    if (nghosts_in_target_[p] == 0) \
                        nghosts_in_target_[p] = 1; \
                    new_nghosts += nghosts_in_target_[p]; \
                } \
                nghosts_ = new_nghosts; \
            } \
            sbuf_ = new char[max_size_]; \
            rbuf_ = new char[max_size_]; \
            sreq_ = new MPI_Request[nghosts_]; \
            rreq_ = new MPI_Request[nghosts_]; \
            /* for large graphs, if iteration counts are not reduced it takes >> time */\
            if (lne > 1000) \
            { \
                if (bw_loop_count_ == BW_LOOP_COUNT) \
                    bw_loop_count_ = bw_loop_count_large_; \
                if (bw_skip_count_ == BW_SKIP_COUNT) \
                    bw_skip_count_ = bw_skip_count_large_; \
            } \
        } while(0)

        Comm(Graph* g):
            g_(g), nghosts_(0), lnv_(0),
            pindex_(0), nghosts_in_target_(0),
            sbuf_(nullptr), rbuf_(nullptr),
            sreq_(nullptr), rreq_(nullptr),
            max_size_(MAX_SIZE), min_size_(MIN_SIZE),
            large_msg_size_(LARGE_SIZE),
            bw_loop_count_(BW_LOOP_COUNT), 
            bw_loop_count_large_(BW_LOOP_COUNT_LARGE),
            bw_skip_count_(BW_SKIP_COUNT), 
            bw_skip_count_large_(BW_SKIP_COUNT_LARGE),
            lt_loop_count_(LT_LOOP_COUNT), 
            lt_loop_count_large_(LT_LOOP_COUNT_LARGE),
            lt_skip_count_(LT_SKIP_COUNT), 
            lt_skip_count_large_(LT_SKIP_COUNT_LARGE),
            targets_(0), degree_(0)
        { COMM_COMMON(); }
         
        Comm(Graph* g, GraphElem min_size, GraphElem max_size, int shrink_percent):
            g_(g), nghosts_(0), lnv_(0),
            pindex_(0), nghosts_in_target_(0), 
            sbuf_(nullptr), rbuf_(nullptr),
            sreq_(nullptr), rreq_(nullptr),
            min_size_(min_size), max_size_(max_size), 
            large_msg_size_(LARGE_SIZE), 
            bw_loop_count_(BW_LOOP_COUNT), 
            bw_loop_count_large_(BW_LOOP_COUNT_LARGE),
            bw_skip_count_(BW_SKIP_COUNT), 
            bw_skip_count_large_(BW_SKIP_COUNT_LARGE),
            lt_loop_count_(LT_LOOP_COUNT), 
            lt_loop_count_large_(LT_LOOP_COUNT_LARGE),
            lt_skip_count_(LT_SKIP_COUNT), 
            lt_skip_count_large_(LT_SKIP_COUNT_LARGE),
            targets_(0), degree_(0), shrinkp_(shrink_percent)
        { COMM_COMMON(); }       
        
        Comm(Graph* g, 
                GraphElem max_size, GraphElem min_size,
                GraphElem large_msg_size,
                int bw_loop_count, int bw_loop_count_large,
                int bw_skip_count, int bw_skip_count_large,
                int lt_loop_count, int lt_loop_count_large,
                int lt_skip_count, int lt_skip_count_large):
            g_(g), nghosts_(0), lnv_(0), 
            pindex_(0), nghosts_in_target_(0), 
            sbuf_(nullptr), rbuf_(nullptr),
            sreq_(nullptr), rreq_(nullptr),
            max_size_(max_size), min_size_(min_size),
            large_msg_size_(large_msg_size),
            bw_loop_count_(bw_loop_count), 
            bw_loop_count_large_(bw_loop_count_large),
            bw_skip_count_(bw_skip_count), 
            bw_skip_count_large_(bw_skip_count_large),
            lt_loop_count_(lt_loop_count), 
            lt_loop_count_large_(lt_loop_count_large),
            lt_skip_count_(lt_skip_count), 
            lt_skip_count_large_(lt_skip_count_large),
            targets_(0), degree_(0)
        { COMM_COMMON(); }
        
        ~Comm() 
        {            
            targets_.clear();
            pindex_.clear();
            nghosts_in_target_.clear();
            delete []sbuf_;
            delete []rbuf_;
            delete []sreq_;
            delete []rreq_;
        }

        void touch_buffers(GraphElem const& size)
        {
            std::memset(sbuf_, 'a', size);
            std::memset(rbuf_, 'b', size);
        }

        // kernel for bandwidth 
        // (extra s/w overhead for determining 
        // owner and accessing CSR)
        inline void comm_kernel_bw_extra_overhead(GraphElem const& size)
        {
            // prepost recvs
            for (GraphElem g = 0; g < nghosts_; g++)
            {
                MPI_Irecv(rbuf_, size, MPI_CHAR, MPI_ANY_SOURCE, 
                        100, comm_, rreq_ + g);
            }

            // sends
            GraphElem ng = 0;
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);

                for (GraphElem e = e0; e < e1; e++)
                {
                    Edge const& edge = g_->get_edge(e);
                    const int owner = g_->get_owner(edge.tail_); 
                    if (owner != rank_)
                    {
                        MPI_Isend(sbuf_, size, MPI_CHAR, owner, 
                                100, comm_, sreq_+ ng);
                        ng++;
                    }
                }
            }

            MPI_Waitall(nghosts_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(nghosts_, sreq_, MPI_STATUSES_IGNORE);
        }

        // kernel for bandwidth 
        inline void comm_kernel_bw(GraphElem const& size)
        {
            // prepost recvs
            for (GraphElem g = 0; g < nghosts_; g++)
            {
                MPI_Irecv(rbuf_, size, MPI_CHAR, MPI_ANY_SOURCE, 
                        100, comm_, rreq_ + g);
            }

            // sends
            GraphElem ng = 0;
            for (int p : targets_)
            {
                for (GraphElem g = 0; g < nghosts_in_target_[pindex_[p]]; g++)
                {
                    MPI_Isend(sbuf_, size, MPI_CHAR, p, 100, comm_, sreq_+ ng);
                    ng++;
                }
            }

            MPI_Waitall(nghosts_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(nghosts_, sreq_, MPI_STATUSES_IGNORE);
        }

        // kernel for latency
        inline void comm_kernel_lt(GraphElem const& size)
        { 
            for (int p = 0; p < degree_; p++)
            {
                MPI_Irecv(rbuf_, size, MPI_CHAR, targets_[p], 
                        100, comm_, rreq_ + p);
            }
            
            for (int p = 0; p < degree_; p++)
            {
                MPI_Isend(sbuf_, size, MPI_CHAR, targets_[p], 
                        100, comm_, sreq_ + p);
            }

            MPI_Waitall(degree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(degree_, sreq_, MPI_STATUSES_IGNORE);
        }

        // kernel for latency with extra input parameters
        inline void comm_kernel_lt(GraphElem const& size, GraphElem const& npairs, 
                MPI_Comm gcomm, int const& me)
        { 
            for (int p = 0, j = 0; p < npairs; p++)
            {
                if (p != me)
                {
                    MPI_Irecv(rbuf_, size, MPI_CHAR, p, 
                            100, gcomm, rreq_ + (j++));
                }
            }

            for (int p = 0, j = 0; p < npairs; p++)
            {
                if (p != me)
                {
                    MPI_Isend(sbuf_, size, MPI_CHAR, p, 
                            100, gcomm, sreq_ + (j++));
                }
            }

            MPI_Waitall(npairs-1, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(npairs-1, sreq_, MPI_STATUSES_IGNORE);
        }
        
        // kernel for bandwidth with extra input parameters
        inline void comm_kernel_bw(GraphElem const& size, GraphElem const& npairs, 
                MPI_Comm gcomm, GraphElem const& avg_ng, int const& me)
        {
            // prepost recvs
            for (int p = 0, j = 0; p < npairs; p++)
            {
                if (p != me)
                {
                    for (GraphElem g = 0; g < avg_ng; g++)
                    {
                        MPI_Irecv(rbuf_, size, MPI_CHAR, MPI_ANY_SOURCE, 
                                100, gcomm, rreq_ + (j++));
                    }
                }
            }

            // sends
            for (int p = 0, j = 0; p < npairs; p++)
            {
                if (p != me)
                {
                    for (GraphElem g = 0; g < avg_ng; g++)
                    {
                        MPI_Isend(sbuf_, size, MPI_CHAR, p, 
                                100, gcomm, sreq_+ (j++));
                    }
                }
            }

            MPI_Waitall(avg_ng*(npairs-1), rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(avg_ng*(npairs-1), sreq_, MPI_STATUSES_IGNORE);
        }

        // Bandwidth test
        void p2p_bw()
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = bw_loop_count_, skip = bw_skip_count_;
            
            // find average number of ghost vertices
            GraphElem sum_ng = 0, avg_ng;
            MPI_Reduce(&nghosts_, &sum_ng, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
            avg_ng = sum_ng / size_;

            // total communicating pairs
            int sum_npairs = 0;
            MPI_Reduce(&degree_, &sum_npairs, 1, MPI_INT, MPI_SUM, 0, comm_);
            
            if(rank_ == 0) 
            {
                std::cout << "--------------------------------" << std::endl;
                std::cout << "--------Bandwidth test----------" << std::endl;
                std::cout << "--------------------------------" << std::endl;
                std::cout << std::setw(12) << "# Bytes" << std::setw(13) << "MB/s" 
                    << std::setw(13) << "Msg/s" 
                    << std::setw(18) << "Variance" 
                    << std::setw(15) << "STDDEV" 
                    << std::setw(16) << "95% CI" 
                    << std::endl;
            }

            for (GraphElem size = (!min_size_ ? 1 : min_size_); size <= max_size_; size *= 2) 
            {
                // memset
                touch_buffers(size);

                if(size > large_msg_size_) 
                {
                    loop = bw_loop_count_large_;
                    skip = bw_skip_count_large_;
                }
    
                MPI_Barrier(comm_);

                // time communication kernel
                for (int l = 0; l < loop + skip; l++) 
                {           
                    if (l == skip)
                    {
                        MPI_Barrier(comm_);
                        t_start = MPI_Wtime();
                    }

                    comm_kernel_bw(size);
                }   

                t_end = MPI_Wtime();
                t = t_end - t_start;

                // execution time stats
                MPI_Reduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_t = sum_t / size_;
                double t_sq = t*t;
                double sum_tsq = 0;
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_tsq = sum_tsq / size_;
                double var = avg_tsq - (avg_t*avg_t);
                double stddev = sqrt(var);

                MPI_Barrier(comm_);
                
                if (rank_ == 0) 
                {
                    double tmp = size / 1e6 * loop * avg_ng;
                    sum_t /= sum_npairs;
                    double bw = tmp / sum_t;

                    std::cout << std::setw(10) << size << std::setw(15) << bw 
                        << std::setw(15) << 1e6 * bw / size
                        << std::setw(18) << var
                        << std::setw(16) << stddev 
                        << std::setw(16) << stddev * ZCI / sqrt((double)(loop * avg_ng)) 
                        << std::endl;
                }
            }
        }
         
        // Latency test
        void p2p_lt()
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = lt_loop_count_, skip = lt_skip_count_;
            
            // total communicating pairs
            int sum_npairs = 0;
            MPI_Reduce(&degree_, &sum_npairs, 1, MPI_INT, MPI_SUM, 0, comm_);

            if(rank_ == 0) 
            {
                std::cout << "--------------------------------" << std::endl;
                std::cout << "----------Latency test----------" << std::endl;
                std::cout << "--------------------------------" << std::endl;
                std::cout << std::setw(12) << "# Bytes" << std::setw(15) << "Lat(us)" 
                    << std::setw(16) << "Variance" 
                    << std::setw(15) << "STDDEV" 
                    << std::setw(16) << "95% CI" 
                    << std::endl;
            }

            for (GraphElem size = min_size_; size <= max_size_; size  = (size ? size * 2 : 1))
            {       
                // memset
                touch_buffers(size);

                if(size > large_msg_size_) 
                {
                    loop = lt_loop_count_large_;
                    skip = lt_skip_count_large_;
                }
                        
                MPI_Barrier(comm_);

                // time communication kernel
                for (int l = 0; l < loop + skip; l++) 
                {           
                    if (l == skip)
                    {
                        t_start = MPI_Wtime();
                        MPI_Barrier(comm_);
                    }
                    
                    comm_kernel_lt(size);
                }   

                t_end = MPI_Wtime();
                t = (t_end - t_start); 

                // execution time stats
                MPI_Reduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
                double t_sq = t*t;
                double sum_tsq = 0;
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_st = sum_t / (double) size_;
                double avg_tsq   = sum_tsq / (double) size_;
                double var = avg_tsq - (avg_st*avg_st);
                double stddev  = sqrt(var);
                
                sum_t *= 1.0e6 / loop;
                double avg_t = sum_t / (double)sum_npairs;
                
                MPI_Barrier(comm_);
                
                if (rank_ == 0) 
                {
                    std::cout << std::setw(10) << size << std::setw(17) << avg_t
                        << std::setw(16) << var
                        << std::setw(16) << stddev 
                        << std::setw(16) << stddev * ZCI / sqrt(loop) 
                        << std::endl;
                }
            }
        }     

        // Bandwidth/Latency estimation by analyzing a 
        // single process neighborhood
        
        // Bandwidth test (single neighborhood)
        void p2p_bw_snbr(int target_nbrhood, GraphElem max_ng = -1)
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = bw_loop_count_, skip = bw_skip_count_;
            assert(target_nbrhood < size_);
             
            // find average number of ghost vertices
            GraphElem avg_ng, sum_ng = nghosts_;
            MPI_Allreduce(MPI_IN_PLACE, &sum_ng, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
            avg_ng = sum_ng / size_;
            
            if (max_ng > 0)
            {
                avg_ng = std::min(avg_ng, max_ng);
                if (rank_ == 0)
                {
                    std::cout << "Number of ghost vertices set as: " << avg_ng << std::endl;
                }
            }

            // extract process neighborhood of target_nbrhood PE
            int tgt_deg = degree_;
            int tgt_rank = MPI_UNDEFINED, tgt_size = 0;
            MPI_Bcast(&tgt_deg, 1, MPI_INT, target_nbrhood, comm_);

            std::vector<int> exl_tgt(tgt_deg+1);
            if (rank_ == target_nbrhood)
                std::copy(targets_.begin(), targets_.end(), exl_tgt.begin());
            
            MPI_Bcast(exl_tgt.data(), tgt_deg, MPI_INT, target_nbrhood, comm_);
            exl_tgt[tgt_deg] = target_nbrhood;

            // create new group/comm
            MPI_Group cur_grp, nbr_grp;
            MPI_Comm nbr_comm;

            MPI_Comm_group(comm_, &cur_grp);
            MPI_Group_incl(cur_grp, tgt_deg+1, exl_tgt.data(), &nbr_grp);
            MPI_Comm_create(comm_, nbr_grp, &nbr_comm);
            
            MPI_Group_rank(nbr_grp, &tgt_rank);
            MPI_Group_size(nbr_grp, &tgt_size);

            if(rank_ == target_nbrhood) 
            {
                std::cout << "------------------------------------------" << std::endl;
                std::cout << "---Bandwidth test (single neighborhood)---" << std::endl;
                std::cout << "------------------------------------------" << std::endl;
                std::cout << std::setw(12) << "# Bytes" << std::setw(13) << "MB/s" 
                    << std::setw(13) << "Msg/s" 
                    << std::setw(18) << "Variance" 
                    << std::setw(15) << "STDDEV" 
                    << std::setw(16) << "95% CI" 
                    << std::endl;
            }

            // start communication only if belongs to the 
            // chosen process neighborhood 
            if (tgt_rank != MPI_UNDEFINED)
            {
                // readjust request buffer sizes and counts
                delete []sreq_;
                delete []rreq_;
                sreq_ = new MPI_Request[tgt_deg*avg_ng];
                rreq_ = new MPI_Request[tgt_deg*avg_ng];
                              
                if (bw_loop_count_ == BW_LOOP_COUNT_LARGE)
                    bw_loop_count_ = BW_LOOP_COUNT;
                if (bw_skip_count_ == BW_SKIP_COUNT_LARGE)
                    bw_skip_count_ = BW_SKIP_COUNT;

                for (GraphElem size = (!min_size_ ? 1 : min_size_); size <= max_size_; size *= 2) 
                {
                    // memset
                    touch_buffers(size);

                    if(size > large_msg_size_) 
                    {
                        loop = bw_loop_count_large_;
                        skip = bw_skip_count_large_;
                    }

                    MPI_Barrier(nbr_comm);

                    // time communication kernel
                    for (int l = 0; l < loop + skip; l++) 
                    {           
                        if (l == skip)
                        {
                            MPI_Barrier(nbr_comm);
                            t_start = MPI_Wtime();
                        }
        
                        comm_kernel_bw(size, tgt_deg+1, nbr_comm, avg_ng, tgt_rank);
                    }   

                    t_end = MPI_Wtime();
                    t = t_end - t_start;

                    // execution time stats
                    MPI_Reduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, nbr_comm);

                    double avg_t = sum_t / tgt_size;
                    double t_sq = t*t;
                    double sum_tsq = 0;
                    MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, nbr_comm);

                    double avg_tsq   = sum_tsq / tgt_size;
                    double var = avg_tsq - (avg_t*avg_t);
                    double stddev  = sqrt(var);

                    MPI_Barrier(nbr_comm);

                    if (tgt_rank == 0) 
                    {            
                        double tmp = size / 1e6 * loop * avg_ng * 2.0;
                        sum_t /= (tgt_size * 2.0);
                        double bw = tmp / sum_t;
                    
                        std::cout << std::setw(10) << size << std::setw(15) << bw 
                            << std::setw(15) << 1e6 * bw / size
                            << std::setw(18) << var
                            << std::setw(16) << stddev 
                            << std::setw(16) << stddev * ZCI / sqrt((double)(loop * avg_ng * 2.0)) 
                            << std::endl;
                    }
                }
            }

            // remaining processes wait on 
            // a barrier
            MPI_Barrier(comm_);
            
            if (nbr_comm != MPI_COMM_NULL)
                MPI_Comm_free(&nbr_comm);
 
            MPI_Group_free(&cur_grp);
            MPI_Group_free(&nbr_grp);
        }
         
        // Latency test (single neighborhood)
        void p2p_lt_snbr(int target_nbrhood)
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = lt_loop_count_, skip = lt_skip_count_;
            int tgt_rank = MPI_UNDEFINED, tgt_size = 0;
            
            assert(target_nbrhood < size_);

            // total communicating pairs
            int sum_npairs = degree_;
            MPI_Allreduce(MPI_IN_PLACE, &sum_npairs, 1, MPI_INT, MPI_SUM, comm_);

            // extract process neighborhood of target_nbrhood PE
            int tgt_deg = degree_;
            MPI_Bcast(&tgt_deg, 1, MPI_INT, target_nbrhood, comm_);

            std::vector<int> exl_tgt(tgt_deg+1);
            if (rank_ == target_nbrhood)
                std::copy(targets_.begin(), targets_.end(), exl_tgt.begin());
            
            MPI_Bcast(exl_tgt.data(), tgt_deg, MPI_INT, target_nbrhood, comm_);
            exl_tgt[tgt_deg] = target_nbrhood;

            // create new group/comm
            MPI_Group cur_grp, nbr_grp;
            MPI_Comm nbr_comm;

            MPI_Comm_group(comm_, &cur_grp);
            MPI_Group_incl(cur_grp, tgt_deg+1, exl_tgt.data(), &nbr_grp);
            MPI_Comm_create(comm_, nbr_grp, &nbr_comm);
            
            MPI_Group_rank(nbr_grp, &tgt_rank);
            MPI_Group_size(nbr_grp, &tgt_size);

            if(rank_ == target_nbrhood) 
            {
                std::cout << "------------------------------------------" << std::endl;
                std::cout << "----Latency test (single neighborhood)----" << std::endl;
                std::cout << "------------------------------------------" << std::endl;
                std::cout << std::setw(12) << "# Bytes" << std::setw(15) << "Lat(us)" 
                    << std::setw(16) << "Variance" 
                    << std::setw(15) << "STDDEV" 
                    << std::setw(16) << "95% CI" 
                    << std::endl;
            }

            // start communication only if belongs to the 
            // chosen process neighborhood 
            if (tgt_rank != MPI_UNDEFINED)
            {
                for (GraphElem size = min_size_; size <= max_size_; size  = (size ? size * 2 : 1))
                {       
                    // memset
                    touch_buffers(size);

                    if(size > large_msg_size_) 
                    {
                        loop = lt_loop_count_large_;
                        skip = lt_skip_count_large_;
                    }

                    MPI_Barrier(nbr_comm);

                    // time communication kernel
                    for (int l = 0; l < loop + skip; l++) 
                    {           
                        if (l == skip)
                        {
                            t_start = MPI_Wtime();
                            MPI_Barrier(nbr_comm);
                        }
                        
                        comm_kernel_lt(size, tgt_deg+1, nbr_comm, tgt_rank);
                    }   

                    t_end = MPI_Wtime();
                    t = (t_end - t_start);

                    // execution time stats
                    MPI_Reduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, nbr_comm);

                    double t_sq = t*t;
                    double sum_tsq = 0;
                    MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, nbr_comm);

                    double avg_st = sum_t / tgt_size;
                    double avg_tsq   = sum_tsq / tgt_size;
                    double var = avg_tsq - (avg_st*avg_st);
                    double stddev  = sqrt(var);

                    sum_t *= 1.0e6 / (double)(loop*2.0);
                    double avg_t = sum_t / (double)(tgt_size*2.0);
                    
                    MPI_Barrier(nbr_comm);

                    if (tgt_rank == 0) 
                    {
                        std::cout << std::setw(10) << size << std::setw(17) << avg_t
                            << std::setw(16) << var
                            << std::setw(16) << stddev 
                            << std::setw(16) << stddev * ZCI / sqrt(loop * 2.0) 
                            << std::endl;
                    }
                }
            }

            // remaining processes wait on 
            // a barrier
            MPI_Barrier(comm_);

            if (nbr_comm != MPI_COMM_NULL)
                MPI_Comm_free(&nbr_comm);
            
            MPI_Group_free(&cur_grp);
            MPI_Group_free(&nbr_grp);
        }      

    private:
        Graph* g_;
        GraphElem nghosts_, lnv_;
        // ghost vertices in target rank
        std::vector<GraphElem> nghosts_in_target_; 
        std::unordered_map<int, int> pindex_; 

        char *sbuf_, *rbuf_;
        MPI_Request *sreq_, *rreq_;

        // ranges
        GraphElem max_size_, min_size_, large_msg_size_;
        int bw_loop_count_, bw_loop_count_large_,
            bw_skip_count_, bw_skip_count_large_,
            lt_loop_count_, lt_loop_count_large_,
            lt_skip_count_, lt_skip_count_large_;

        int shrinkp_; // graph shrink percent
        int rank_, size_, degree_;
        std::vector<int> targets_;
        MPI_Comm comm_;
};

#endif
