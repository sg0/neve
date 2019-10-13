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

#if defined(SCOREP_USER_ENABLE)
#include <scorep/SCOREP_User.h>
#endif

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

class Comm
{
    public:

#define COMM_COMMON() \
        do { \
            comm_ = g_->get_comm(); \
            MPI_Comm_size(comm_, &size_); \
            MPI_Comm_rank(comm_, &rank_); \
            const GraphElem lne = g_->get_lne(); \
            lnv_ = g_->get_lnv(); \
            std::vector<GraphElem> a2a_send_dat(size_), a2a_recv_dat(size_); \
            /* track outgoing ghosts not owned by me */ \
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
                        if (std::find(targets_.begin(), targets_.end(), owner) == targets_.end()) \
                        { \
                            targets_.push_back(owner); \
                            target_pindex_.insert({owner, outdegree_}); \
                            outdegree_++; \
                            nghosts_in_target_.push_back(0); \
                        } \
                        out_nghosts_++; \
                        a2a_send_dat[owner]++; \
                        nghosts_in_target_[target_pindex_[owner]]++; \
                    } \
                } \
            } \
            assert(outdegree_ == nghosts_in_target_.size()); \
            if (shrinkp_ > 0.0) \
            { \
                GraphElem new_nghosts = 0; \
                std::unordered_map<int, int>::iterator peit = target_pindex_.begin(); \
                for (int p = 0; p < outdegree_; p++) \
                { \
                    nghosts_in_target_[p] = (int)((shrinkp_ * (float)nghosts_in_target_[p]) / (float)100); \
                    if (nghosts_in_target_[p] == 0) \
                        nghosts_in_target_[p] = 1; \
                    new_nghosts += nghosts_in_target_[p]; \
                    a2a_send_dat[peit->first] = nghosts_in_target_[p]; \
                    ++peit; \
                } \
                GraphElem nghosts[2] = {out_nghosts_, new_nghosts}, all_nghosts[2] = {0, 0}; \
                MPI_Reduce(nghosts, all_nghosts, 2, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_); \
                out_nghosts_ = new_nghosts; \
                if (rank_ == 0) \
                { \
                    std::cout << "Considering only " << shrinkp_ << "% of overall #ghosts, previous total outgoing #ghosts: " \
                    << all_nghosts[0] << ", current total outgoing #ghosts: " << all_nghosts[1] << std::endl; \
                } \
            } \
            /* track incoming communication (processes for which I am a ghost) */ \
            /* send to PEs in targets_ list about shared ghost info */ \
            MPI_Alltoall(a2a_send_dat.data(), 1, MPI_GRAPH_TYPE, a2a_recv_dat.data(), 1, MPI_GRAPH_TYPE, comm_); \
            MPI_Barrier(comm_); \
            for (int p = 0; p < size_; p++) \
            { \
                if (a2a_recv_dat[p] > 0) \
                { \
                    source_pindex_.insert({p, indegree_}); \
                    sources_.push_back(p); \
                    nghosts_in_source_.push_back(a2a_recv_dat[p]); \
                    indegree_++; \
                    in_nghosts_ += a2a_recv_dat[p]; \
                } \
            } \
            assert(indegree_ == nghosts_in_source_.size()); \
            sbuf_ = new char[out_nghosts_*max_size_]; \
            rbuf_ = new char[in_nghosts_*max_size_]; \
            assert(in_nghosts_ >= indegree_); \
            assert(out_nghosts_ >= outdegree_); \
            sreq_ = new MPI_Request[out_nghosts_]; \
            rreq_ = new MPI_Request[in_nghosts_]; \
            /* for large graphs, if iteration counts are not reduced it takes >> time */\
	    if (lne > 1000) \
            { \
                if (bw_loop_count_ == BW_LOOP_COUNT) \
                    bw_loop_count_ = bw_loop_count_large_; \
                if (bw_skip_count_ == BW_SKIP_COUNT) \
                    bw_skip_count_ = bw_skip_count_large_; \
            } \
	    a2a_send_dat.clear(); \
            a2a_recv_dat.clear(); \
        } while(0)

        Comm(Graph* g):
            g_(g), in_nghosts_(0), out_nghosts_(0), lnv_(0),
            target_pindex_(0), source_pindex_(0),
            nghosts_in_target_(0), nghosts_in_source_(0),
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
            targets_(0), sources_(0), indegree_(0), outdegree_(0)
        { COMM_COMMON(); }
         
        Comm(Graph* g, GraphElem min_size, GraphElem max_size, float shrink_percent):
            g_(g), in_nghosts_(0), out_nghosts_(0), lnv_(0),
            target_pindex_(0), source_pindex_(0),
            nghosts_in_target_(0), nghosts_in_source_(0),
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
            targets_(0), sources_(0), indegree_(0), outdegree_(0),
            shrinkp_(shrink_percent)
        { COMM_COMMON(); }       
        
        Comm(Graph* g, 
                GraphElem max_size, GraphElem min_size,
                GraphElem large_msg_size,
                int bw_loop_count, int bw_loop_count_large,
                int bw_skip_count, int bw_skip_count_large,
                int lt_loop_count, int lt_loop_count_large,
                int lt_skip_count, int lt_skip_count_large):
            g_(g), in_nghosts_(0), out_nghosts_(0), lnv_(0),
            target_pindex_(0), source_pindex_(0),
            nghosts_in_target_(0), nghosts_in_source_(0),
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
            targets_(0), sources_(0), indegree_(0), outdegree_(0)
        { COMM_COMMON(); }
        
        ~Comm() 
        {            
            targets_.clear();
            target_pindex_.clear();
            nghosts_in_target_.clear();
            sources_.clear();
            source_pindex_.clear();
            nghosts_in_source_.clear();
            delete []sbuf_;
            delete []rbuf_;
            delete []sreq_;
            delete []rreq_;
        }

        void touch_buffers(GraphElem const& size)
        { 
            std::memset(sbuf_, 'a', out_nghosts_*size); 
            std::memset(rbuf_, 'b', in_nghosts_*size); 
        }
        
        // kernel for bandwidth 
        // (extra s/w overhead for determining 
        // owner and accessing CSR)
        inline void comm_kernel_bw_extra_overhead(GraphElem const& size)
        {
            // prepost recvs
            for (GraphElem g = 0; g < in_nghosts_; g++)
            {
                MPI_Irecv(&rbuf_[g*size], size, MPI_CHAR, sources_[g], 
                        g, comm_, rreq_ + g);
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
                        MPI_Isend(&sbuf_[ng*size], size, MPI_CHAR, owner, 
                                ng, comm_, sreq_+ ng);
                        ng++;
                    }
                }
            }

            MPI_Waitall(in_nghosts_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(out_nghosts_, sreq_, MPI_STATUSES_IGNORE);
        }
       
        // kernel for latency
        inline void comm_kernel_lt(GraphElem const& size)
	{
	    for (int p = 0; p < indegree_; p++)
	    {
		MPI_Irecv(rbuf_, size, MPI_CHAR, sources_[p], 100, comm_, rreq_ + p);
	    }

	    for (int p = 0; p < outdegree_; p++)
	    {
		MPI_Isend(sbuf_, size, MPI_CHAR, targets_[p], 100, comm_, sreq_ + p);
	    }

            MPI_Waitall(indegree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
        }

        // kernel for latency with extra input parameters
        inline void comm_kernel_lt(GraphElem const& size, GraphElem const& npairs, 
                MPI_Comm gcomm, int const& me)
        { 
            for (int p = 0, j = 0; p < npairs; p++)
            {
                if (p != me)
                {
                    MPI_Irecv(rbuf_, size, MPI_CHAR, p, 100, gcomm, rreq_ + j);
                    j++;
                }
            }

            for (int p = 0, j = 0; p < npairs; p++)
            {
                if (p != me)
                {
                    MPI_Isend(sbuf_, size, MPI_CHAR, p, 100, gcomm, sreq_ + j);
                    j++;
                }
            }

            MPI_Waitall(npairs-1, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(npairs-1, sreq_, MPI_STATUSES_IGNORE);
        }
         
	// kernel for bandwidth 
        inline void comm_kernel_bw(GraphElem const& size)
        {
            GraphElem rng = 0, sng = 0;

            // prepost recvs
            for (int p = 0; p < indegree_; p++)
            {
                for (GraphElem g = 0; g < nghosts_in_source_[p]; g++)
                {
                    MPI_Irecv(&rbuf_[rng*size], size, MPI_CHAR, sources_[p], g, comm_, rreq_ + rng);
                    rng++;
                }
            }

            // sends
            for (int p = 0; p < outdegree_; p++)
            {
                for (GraphElem g = 0; g < nghosts_in_target_[p]; g++)
                {
                    MPI_Isend(&sbuf_[sng*size], size, MPI_CHAR, targets_[p], g, comm_, sreq_+ sng);
                    sng++;
                }
            }

            MPI_Waitall(in_nghosts_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(out_nghosts_, sreq_, MPI_STATUSES_IGNORE);
        }
        
        // kernel for bandwidth with extra input parameters and 
        // out of order receives
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
                        MPI_Irecv(&rbuf_[j*size], size, MPI_CHAR, MPI_ANY_SOURCE, j, gcomm, rreq_ + j);
                        j++;
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
                        MPI_Isend(&sbuf_[j*size], size, MPI_CHAR, p, j, gcomm, sreq_+ j);
                        j++;
                    }
                }
            }

            MPI_Waitall(avg_ng*(npairs-1), rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(avg_ng*(npairs-1), sreq_, MPI_STATUSES_IGNORE);
        }

        // Bandwidth tests
        void p2p_bw()
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = bw_loop_count_, skip = bw_skip_count_;
            
            // total communicating pairs
            int sum_npairs = outdegree_ + indegree_;
            MPI_Allreduce(MPI_IN_PLACE, &sum_npairs, 1, MPI_INT, MPI_SUM, comm_);
            sum_npairs /= 2;
             
            // find average number of ghost vertices
            GraphElem sum_ng = out_nghosts_ + in_nghosts_, avg_ng;
            MPI_Allreduce(MPI_IN_PLACE, &sum_ng, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
            avg_ng = sum_ng / sum_npairs;
           
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
    
#if defined(SCOREP_USER_ENABLE)
	        SCOREP_RECORDING_ON();
#endif
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

#if defined(SCOREP_USER_ENABLE)
		SCOREP_RECORDING_OFF();
#endif
                t_end = MPI_Wtime();
                t = t_end - t_start;

                // execution time stats
                MPI_Reduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_t = sum_t / sum_npairs;
                double avg_st = sum_t / size_;
                double t_sq = t*t;
                double sum_tsq = 0;
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_tsq = sum_tsq / size_;
                double var = avg_tsq - (avg_st*avg_st);
                double stddev = sqrt(var);

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
         
        // no extra loop, just communication among ghosts
        void p2p_bw_hardskip()
        {
            double t, t_start, t_end, sum_t = 0.0;
            
            // total communicating pairs
            int sum_npairs = outdegree_ + indegree_;
            MPI_Allreduce(MPI_IN_PLACE, &sum_npairs, 1, MPI_INT, MPI_SUM, comm_);
            sum_npairs /= 2;
             
            // find average number of ghost vertices
            GraphElem sum_ng = out_nghosts_, avg_ng;
            MPI_Allreduce(MPI_IN_PLACE, &sum_ng, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
            avg_ng = sum_ng / sum_npairs;
           
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

                MPI_Barrier(comm_);

                t_start = MPI_Wtime();

                comm_kernel_bw(size);

                t_end = MPI_Wtime();
                t = t_end - t_start;

                // execution time stats
                MPI_Reduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_t = sum_t / sum_npairs;
                double avg_st = sum_t / size_;
                double t_sq = t*t;
                double sum_tsq = 0;
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

                double avg_tsq = sum_tsq / size_;
                double var = avg_tsq - (avg_st*avg_st);
                double stddev = sqrt(var);

                if (rank_ == 0) 
                {
		    double tmp = size / 1e6 * avg_ng;
                    sum_t /= sum_npairs;
                    double bw = tmp / sum_t;

                    std::cout << std::setw(10) << size << std::setw(15) << bw 
                        << std::setw(15) << 1e6 * bw / size
                        << std::setw(18) << var
                        << std::setw(16) << stddev 
                        << std::setw(16) << stddev * ZCI / sqrt((double)(avg_ng)) 
                        << std::endl;
                }
            }
        }
        
        // Latency test
        void p2p_lt()
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = lt_loop_count_, skip = lt_skip_count_;
            
            std::vector<double> plat(size_);
            int n99 = (int)std::ceil(0.99*size_);

            // total communicating pairs
            int sum_npairs = outdegree_ + indegree_;
            MPI_Allreduce(MPI_IN_PLACE, &sum_npairs, 1, MPI_INT, MPI_SUM, comm_);
            sum_npairs /= 2;

            if(rank_ == 0) 
            {
                std::cout << "--------------------------------" << std::endl;
                std::cout << "----------Latency test----------" << std::endl;
                std::cout << "--------------------------------" << std::endl;
                std::cout << std::setw(12) << "# Bytes" << std::setw(15) << "Lat(us)" 
                    << std::setw(16) << "Max(us)" 
                    << std::setw(16) << "99%(us)" 
                    << std::setw(16) << "Variance" 
                    << std::setw(15) << "STDDEV" 
                    << std::setw(16) << "95% CI" 
                    << std::endl;
            }

	    for (GraphElem size = min_size_; size <= max_size_; size  = (size ? size * 2 : 1))
            {       
                MPI_Barrier(comm_);

                if (size > large_msg_size_) 
                {
                    loop = lt_loop_count_large_;
                    skip = lt_skip_count_large_;
		}
                
#if defined(SCOREP_USER_ENABLE)
	        SCOREP_RECORDING_ON();
		SCOREP_USER_REGION_BY_NAME_BEGIN("TRACER_Loop", SCOREP_USER_REGION_TYPE_COMMON);
		if (rank_ == 0)
			SCOREP_USER_REGION_BY_NAME_BEGIN("TRACER_WallTime_MainLoop", SCOREP_USER_REGION_TYPE_COMMON);
#endif
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

#if defined(SCOREP_USER_ENABLE)
		  if (rank_ == 0)
			  SCOREP_USER_REGION_BY_NAME_END("TRACER_WallTime_MainLoop");
		  
		  SCOREP_USER_REGION_BY_NAME_END("TRACER_Loop");
		  SCOREP_RECORDING_OFF();
#endif
                t_end = MPI_Wtime();
                t = (t_end - t_start) * 1.0e6 / (double)loop; 
                double t_sq = t*t;
                double sum_tsq = 0;
                
                // execution time stats
                MPI_Allreduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, comm_);
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

		double avg_t = sum_t / (double) sum_npairs;
		double avg_st = sum_t / (double) size_; // no. of observations
                double avg_tsq = sum_tsq / (double) size_;
                double var = avg_tsq - (avg_st*avg_st);
                double stddev  = sqrt(var);
                
                double lmax = 0.0;
                MPI_Reduce(&t, &lmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
                MPI_Gather(&t, 1, MPI_DOUBLE, plat.data(), 1, MPI_DOUBLE, 0, comm_);
                
                if (rank_ == 0) 
                {
                    std::sort(plat.begin(), plat.end());
                    std::cout << std::setw(10) << size << std::setw(17) << avg_t
                        << std::setw(16) << lmax/2.0
                        << std::setw(16) << plat[n99-1]/2.0
                        << std::setw(16) << var
                        << std::setw(16) << stddev 
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * 2.0) 
                        << std::endl;
                }
            }
            plat.clear();
        } 

        // Bandwidth/Latency estimation by analyzing a 
        // single process neighborhood
        
        // Bandwidth test (single neighborhood)
        void p2p_bw_snbr(int target_nbrhood, GraphElem max_ng = -1)
        {
            double t, t_start, t_end, sum_t = 0.0;
            int loop = bw_loop_count_, skip = bw_skip_count_;
            assert(target_nbrhood < size_);

            // extract process neighborhood of target_nbrhood PE
            int tgt_deg = outdegree_;
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
                // find average number of ghost vertices
                GraphElem avg_ng, sum_ng = out_nghosts_;
                MPI_Allreduce(MPI_IN_PLACE, &sum_ng, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
                avg_ng = sum_ng / tgt_deg; // number of pairs

                // override computed avg_ng
                if (max_ng > 0)
                {
                    avg_ng = std::min(avg_ng, max_ng);
                    if (rank_ == 0)
                    {
                        std::cout << "Number of ghost vertices set as: " << avg_ng << std::endl;
                    }
                }

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

                    double avg_t = sum_t / tgt_deg;
                    double avg_st = sum_t / tgt_size;
                    double t_sq = t*t;
                    double sum_tsq = 0;
                    MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, nbr_comm);

                    double avg_tsq = sum_tsq / tgt_size;
                    double var = avg_tsq - (avg_st*avg_st);
                    double stddev  = sqrt(var);

                    if (tgt_rank == 0) 
                    {            
                        double tmp = size / 1e6 * loop * avg_ng;
                        sum_t /= tgt_size;
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

            // extract process neighborhood of target_nbrhood PE
            int tgt_deg = outdegree_;
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
                    << std::setw(16) << "Max(us)"
                    << std::setw(16) << "99%(us)" 
                    << std::setw(16) << "Variance" 
                    << std::setw(15) << "STDDEV" 
                    << std::setw(16) << "95% CI" 
                    << std::endl;
            }

            // start communication only if belongs to the 
            // chosen process neighborhood 
            if (tgt_rank != MPI_UNDEFINED)
            {
                std::vector<double> plat(tgt_size);
                int n99 = (int)std::ceil(0.99*tgt_size);

                for (GraphElem size = min_size_; size <= max_size_; size  = (size ? size * 2 : 1))
                {       
                    MPI_Barrier(nbr_comm);
                    
                    if(size > large_msg_size_) 
                    {
                        loop = lt_loop_count_large_;
                        skip = lt_skip_count_large_;
                    }

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
                    t = (t_end - t_start) * 1.0e6 / (2.0 * loop); 

                    // execution time stats
                    MPI_Allreduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, nbr_comm);
                    double t_sq = t*t;
                    double sum_tsq = 0;
                    MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, nbr_comm);

                    double avg_t = sum_t / (double)(2.0*tgt_deg);
                    double avg_st = sum_t / (double)(tgt_size);
                    double avg_tsq = sum_tsq / (double)(tgt_size);
                    double var = avg_tsq - (avg_st*avg_st);
                    double stddev  = sqrt(var);

                    double lmax = 0.0;
                    MPI_Reduce(&t, &lmax, 1, MPI_DOUBLE, MPI_MAX, 0, nbr_comm);
                    MPI_Gather(&t, 1, MPI_DOUBLE, plat.data(), 1, MPI_DOUBLE, 0, nbr_comm);

                    if (tgt_rank == 0) 
                    {
                        std::sort(plat.begin(), plat.end());
                        std::cout << std::setw(10) << size << std::setw(17) << avg_t
                            << std::setw(16) << lmax/2.0
                            << std::setw(16) << plat[n99-1]/2
                            << std::setw(16) << var
                            << std::setw(16) << stddev 
                            << std::setw(16) << stddev * ZCI / sqrt((double)loop * 2.0) 
                            << std::endl;
                    }
                }
                plat.clear();
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
        GraphElem in_nghosts_, out_nghosts_, lnv_;
        // ghost vertices in source/target rank
        std::vector<GraphElem> nghosts_in_target_, nghosts_in_source_; 
        std::unordered_map<int, int> target_pindex_, source_pindex_;

        char *sbuf_, *rbuf_;
        MPI_Request *sreq_, *rreq_;

        // ranges
        GraphElem max_size_, min_size_, large_msg_size_;
        int bw_loop_count_, bw_loop_count_large_,
            bw_skip_count_, bw_skip_count_large_,
            lt_loop_count_, lt_loop_count_large_,
            lt_skip_count_, lt_skip_count_large_;

        float shrinkp_; // graph shrink percent
        int rank_, size_, indegree_, outdegree_;
        std::vector<int> targets_, sources_;
        MPI_Comm comm_;
};

#endif
