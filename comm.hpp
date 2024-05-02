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

#define COMM_COMMON(mpi_version) COMM_COMMON_##mpi_version
#define COMM_COMMON_LT(mpi_version) COMM_COMMON_LT_##mpi_version

#define COMM_COMMON_LT_MPI3 \
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
                            outdegree_++; \
                        } \
                        out_nghosts_++; \
                        a2a_send_dat[owner]++; \
                    } \
                } \
            } \
            assert(outdegree_ == targets_.size()); \
            /* track incoming communication (processes for which I am a ghost) */ \
            /* send to PEs in targets_ list about shared ghost info */ \
            MPI_Alltoall(a2a_send_dat.data(), 1, MPI_GRAPH_TYPE, a2a_recv_dat.data(), 1, MPI_GRAPH_TYPE, comm_); \
            MPI_Barrier(comm_); \
            for (int p = 0; p < size_; p++) \
            { \
                if (a2a_recv_dat[p] > 0) \
                { \
                    sources_.push_back(p); \
                    indegree_++; \
                    in_nghosts_ += a2a_recv_dat[p]; \
                } \
            } \
            assert(indegree_ == sources_.size()); \
            sbuf_ = new char[outdegree_*max_size_]; \
            rbuf_ = new char[indegree_*max_size_]; \
            sreq_ = new MPI_Request[outdegree_]; \
            rreq_ = new MPI_Request[indegree_]; \
	    a2a_send_dat.clear(); \
            a2a_recv_dat.clear(); \
            /* create graph topology communicator for neighbor collectives */ \
            MPI_Dist_graph_create_adjacent(comm_, sources_.size(), sources_.data(), \
                    MPI_UNWEIGHTED, targets_.size(), targets_.data(), MPI_UNWEIGHTED, \
                    MPI_INFO_NULL, 0 , &nbr_comm_); \
            /* following is not necessary, just checking */ \
            int weighted, indeg, outdeg; \
            MPI_Dist_graph_neighbors_count(nbr_comm_, &indeg, &outdeg, &weighted); \
            assert(indegree_ == indeg); \
            assert(outdegree_ == outdeg); \
        } while(0)

#define COMM_COMMON_LT_MPI2 \
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
                            outdegree_++; \
                        } \
                        out_nghosts_++; \
                        a2a_send_dat[owner]++; \
                    } \
                } \
            } \
            assert(outdegree_ == targets_.size()); \
            /* track incoming communication (processes for which I am a ghost) */ \
            /* send to PEs in targets_ list about shared ghost info */ \
            MPI_Alltoall(a2a_send_dat.data(), 1, MPI_GRAPH_TYPE, a2a_recv_dat.data(), 1, MPI_GRAPH_TYPE, comm_); \
            MPI_Barrier(comm_); \
            for (int p = 0; p < size_; p++) \
            { \
                if (a2a_recv_dat[p] > 0) \
                { \
                    sources_.push_back(p); \
                    indegree_++; \
                    in_nghosts_ += a2a_recv_dat[p]; \
                } \
            } \
            assert(indegree_ == sources_.size()); \
            sbuf_ = new char[outdegree_*max_size_]; \
            rbuf_ = new char[indegree_*max_size_]; \
            sreq_ = new MPI_Request[outdegree_]; \
            rreq_ = new MPI_Request[indegree_]; \
	    a2a_send_dat.clear(); \
            a2a_recv_dat.clear(); \
        } while(0)


#define COMM_COMMON_MPI3 \
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
            /* create graph topology communicator for neighbor collectives */ \
            MPI_Dist_graph_create_adjacent(comm_, sources_.size(), sources_.data(), \
                    MPI_UNWEIGHTED, targets_.size(), targets_.data(), MPI_UNWEIGHTED, \
                    MPI_INFO_NULL, 0 , &nbr_comm_); \
            /* following is not necessary, just checking */ \
            int weighted, indeg, outdeg; \
            MPI_Dist_graph_neighbors_count(nbr_comm_, &indeg, &outdeg, &weighted); \
            assert(indegree_ == indeg); \
            assert(outdegree_ == outdeg); \
        } while(0)

#define COMM_COMMON_MPI2 \
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

        explicit Comm(Graph* g):
            g_(g), comm_(MPI_COMM_NULL), nbr_comm_(MPI_COMM_NULL), 
            in_nghosts_(0), out_nghosts_(0), lnv_(0),
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
        { 
            #ifdef SSTMAC
            COMM_COMMON(MPI2); 
            posix_memalign((void**)&sbuf_, sysconf(_SC_PAGESIZE), out_nghosts_*max_size_);
            posix_memalign((void**)&rbuf_, sysconf(_SC_PAGESIZE), in_nghosts_*max_size_);
            std::memset(sbuf_, 0, out_nghosts_*max_size_);
            std::memset(rbuf_, 0, in_nghosts_*max_size_);
            #else
            COMM_COMMON(MPI3);
            #endif
        }
         
        explicit Comm(Graph* g, GraphElem min_size, GraphElem max_size, float shrink_percent):
            g_(g), comm_(MPI_COMM_NULL), nbr_comm_(MPI_COMM_NULL), 
            in_nghosts_(0), out_nghosts_(0), lnv_(0),
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
        { 
            #ifdef SSTMAC
            COMM_COMMON(MPI2); 
            posix_memalign((void**)&sbuf_, sysconf(_SC_PAGESIZE), out_nghosts_*max_size_);
            posix_memalign((void**)&rbuf_, sysconf(_SC_PAGESIZE), in_nghosts_*max_size_);
            std::memset(sbuf_, 0, out_nghosts_*max_size_);
            std::memset(rbuf_, 0, in_nghosts_*max_size_);
            #else
            COMM_COMMON(MPI3);
            #endif
        }       
         
        explicit Comm(Graph* g, GraphElem min_size, GraphElem max_size):
            g_(g), comm_(MPI_COMM_NULL), nbr_comm_(MPI_COMM_NULL), 
            in_nghosts_(0), out_nghosts_(0), lnv_(0),
            target_pindex_(0), source_pindex_(0),
            nghosts_in_target_(0), nghosts_in_source_(0),
            sbuf_(nullptr), rbuf_(nullptr),
            sreq_(nullptr), rreq_(nullptr),
            min_size_(min_size), max_size_(max_size), 
            large_msg_size_(LARGE_SIZE), 
            bw_loop_count_(0), 
            bw_loop_count_large_(0),
            bw_skip_count_(0), 
            bw_skip_count_large_(0),
            lt_loop_count_(LT_LOOP_COUNT), 
            lt_loop_count_large_(LT_LOOP_COUNT_LARGE),
            lt_skip_count_(LT_SKIP_COUNT), 
            lt_skip_count_large_(LT_SKIP_COUNT_LARGE),
            targets_(0), sources_(0), indegree_(0), outdegree_(0),
            shrinkp_(0.0)
        { 
            #ifdef SSTMAC
            COMM_COMMON_LT(MPI2); 
            posix_memalign((void**)&sbuf_, sysconf(_SC_PAGESIZE), outdegree_*max_size_);
            posix_memalign((void**)&rbuf_, sysconf(_SC_PAGESIZE), indegree_*max_size_);
            std::memset(sbuf_, 0, outdegree_*max_size_);
            std::memset(rbuf_, 0, indegree_*max_size_);
            #else
            COMM_COMMON_LT(MPI3);
            #endif
        }

        explicit Comm(Graph* g, 
                GraphElem max_size, GraphElem min_size,
                GraphElem large_msg_size,
                int bw_loop_count, int bw_loop_count_large,
                int bw_skip_count, int bw_skip_count_large,
                int lt_loop_count, int lt_loop_count_large,
                int lt_skip_count, int lt_skip_count_large):
            g_(g), comm_(MPI_COMM_NULL), nbr_comm_(MPI_COMM_NULL), 
            in_nghosts_(0), out_nghosts_(0), lnv_(0),
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
        { 
            #ifdef SSTMAC
            COMM_COMMON(MPI2); 
            posix_memalign((void**)&sbuf_, sysconf(_SC_PAGESIZE), out_nghosts_*max_size_);
            posix_memalign((void**)&rbuf_, sysconf(_SC_PAGESIZE), in_nghosts_*max_size_);
            std::memset(sbuf_, 0, out_nghosts_*max_size_);
            std::memset(rbuf_, 0, in_nghosts_*max_size_);
            #else
            COMM_COMMON(MPI3);
            #endif
        }

        // destroy graph topology communicator
        void destroy_nbr_comm() 
        { 
            if (nbr_comm_ != MPI_COMM_NULL)
                MPI_Comm_free(&nbr_comm_); 
        }
        
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
            std::memset(g_->degree_, 0, lnv_*sizeof(GraphWeight));
        }
        
        void touch_buffers_lt(GraphElem const& size)
        { 
            std::memset(sbuf_, 'a', outdegree_*size); 
            std::memset(rbuf_, 'b', indegree_*size);
            std::memset(g_->degree_, 0, lnv_*sizeof(GraphWeight));
        }

        // work functions
        void sumdegree()
        {
#ifdef USE_OPENMP
            #pragma omp parallel for 
#endif
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                for (GraphElem e = e0; e < e1; e++)
                {
                    Edge const& edge = g_->get_edge(e);
                    g_->degree_[i] += edge.weight_;
                }
            }
        }
         
        void maxdegree()
        {
#ifdef USE_OPENMP
            #pragma omp parallel for 
#endif
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                GraphWeight maxdeg = -1.0;
                g_->edge_range(i, e0, e1);
                for (GraphElem e = e0; e < e1; e++)
                {
                    Edge const& edge = g_->get_edge(e);
                    if (maxdeg < edge.weight_)
                        maxdeg = edge.weight_;
                }
                g_->degree_[i] = maxdeg;
            }
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
       
        // kernel for latency using MPI Isend/Irecv
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
         
        // kernel for latency using MPI Isend/Irecv using 
        // nonblocking consensus
        inline void comm_kernel_lt_nbx(GraphElem const& size)
        {
          for (int p = 0; p < indegree_; p++)
          {
            MPI_Irecv(rbuf_, size, MPI_CHAR, sources_[p], 100, comm_, rreq_ + p);
          }

          for (int p = 0; p < outdegree_; p++)
          {
            MPI_Issend(sbuf_, size, MPI_CHAR, targets_[p], 100, comm_, sreq_ + p);
          }
                 
          bool done = false, nbar_active = false; 
          MPI_Request nbar_req = MPI_REQUEST_NULL;
          while (!done)
          {
            if (nbar_active)
            {
              int test_nbar = -1;
              MPI_Test(&nbar_req, &test_nbar, MPI_STATUS_IGNORE);
              done = !test_nbar ? false : true;
            }
            else
            {
              MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
              MPI_Ibarrier(comm_, &nbar_req);
              nbar_active = true;
            }
          }
        }

#if defined(TEST_LT_MPI_PROC_NULL) 
	// same as above, but replaces target with MPI_PROC_NULL to 
        // measure software overhead
	inline void comm_kernel_lt_pnull(GraphElem const& size)
	{
	    for (int p = 0; p < indegree_; p++)
	    {
		MPI_Irecv(rbuf_, size, MPI_CHAR, MPI_PROC_NULL, 100, comm_, rreq_ + p);
	    }


	    for (int p = 0; p < outdegree_; p++)
	    {
		MPI_Isend(sbuf_, size, MPI_CHAR, MPI_PROC_NULL, 100, comm_, sreq_ + p);
	    }

            MPI_Waitall(indegree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
        }
#endif

        // kernel for latency using MPI Isend/Irecv (invokes usleep)
        inline void comm_kernel_lt_usleep(GraphElem const& size)
	{
	    for (int p = 0; p < indegree_; p++)
	    {
		MPI_Irecv(rbuf_, size, MPI_CHAR, sources_[p], 100, comm_, rreq_ + p);
	    }

            usleep(lnv_);
	    
            for (int p = 0; p < outdegree_; p++)
	    {
		MPI_Isend(sbuf_, size, MPI_CHAR, targets_[p], 100, comm_, sreq_ + p);
	    }

            MPI_Waitall(indegree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
        }
         
        // kernel for latency using MPI Isend/Irecv (invokes sumdegree work kernel)
        inline void comm_kernel_lt_worksum(GraphElem const& size)
	{
	    for (int p = 0; p < indegree_; p++)
	    {
		MPI_Irecv(rbuf_, size, MPI_CHAR, sources_[p], 100, comm_, rreq_ + p);
	    }

            sumdegree();
	    
            for (int p = 0; p < outdegree_; p++)
	    {
		MPI_Isend(sbuf_, size, MPI_CHAR, targets_[p], 100, comm_, sreq_ + p);
	    }

            MPI_Waitall(indegree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
        }      
        
        // kernel for latency using MPI Isend/Irecv (invokes maxdegree work kernel)
        inline void comm_kernel_lt_workmax(GraphElem const& size)
	{
	    for (int p = 0; p < indegree_; p++)
	    {
		MPI_Irecv(rbuf_, size, MPI_CHAR, sources_[p], 100, comm_, rreq_ + p);
	    }

            maxdegree();
	    
            for (int p = 0; p < outdegree_; p++)
	    {
		MPI_Isend(sbuf_, size, MPI_CHAR, targets_[p], 100, comm_, sreq_ + p);
	    }

            MPI_Waitall(indegree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
        }

#if defined(TEST_LT_MPI_PROC_NULL) 
	// same as above, but replaces target with MPI_PROC_NULL to 
        // measure software overhead (invokes usleep)
	inline void comm_kernel_lt_pnull_usleep(GraphElem const& size)
	{
	    for (int p = 0; p < indegree_; p++)
	    {
		MPI_Irecv(rbuf_, size, MPI_CHAR, MPI_PROC_NULL, 100, comm_, rreq_ + p);
	    }

            usleep(lnv_);
	    
            for (int p = 0; p < outdegree_; p++)
	    {
		MPI_Isend(sbuf_, size, MPI_CHAR, MPI_PROC_NULL, 100, comm_, sreq_ + p);
	    }

            MPI_Waitall(indegree_, rreq_, MPI_STATUSES_IGNORE);
            MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
        }
#endif

#ifndef SSTMAC
        // kernel for latency using MPI_Neighbor_alltoall
        inline void comm_kernel_lt_ala(GraphElem const& size)
	{ MPI_Neighbor_alltoall(sbuf_, size, MPI_CHAR, rbuf_, size, MPI_CHAR, nbr_comm_); }

        // kernel for latency using MPI_Neighbor_allgather
        inline void comm_kernel_lt_aga(GraphElem const& size)
	{ MPI_Neighbor_allgather(sbuf_, size, MPI_CHAR, rbuf_, size, MPI_CHAR, nbr_comm_); }
#endif

        // kernel for latency with extra input paragathers
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
  	
        // kernel for bandwidth using NBX 
        inline void comm_kernel_bw_nbx(GraphElem const& size)
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

          bool done = false, nbar_active = false; 
          MPI_Request nbar_req = MPI_REQUEST_NULL;
          while (!done)
          {
            if (nbar_active)
            {
              int test_nbar = -1;
              MPI_Test(&nbar_req, &test_nbar, MPI_STATUS_IGNORE);
              done = !test_nbar ? false : true;
            }
            else
            {
              MPI_Waitall(outdegree_, sreq_, MPI_STATUSES_IGNORE);
              MPI_Ibarrier(comm_, &nbar_req);
              nbar_active = true;
            }
          }
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
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * avg_ng) 
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
                        << std::setw(16) << stddev * ZCI / sqrt((double)avg_ng) 
                        << std::endl;
                }
            }
        }
        
        // Latency test using MPI Isend/Irecv
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
                touch_buffers_lt(size);
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
                    
#if defined(TEST_LT_MPI_PROC_NULL) 
                    comm_kernel_lt_pnull(size);
#else
                    comm_kernel_lt(size);
#endif
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
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * sum_npairs) 
                        << std::endl;
                }
            }
            plat.clear();
        } 

        // Latency test using MPI Isend/Irecv, including usleep
        void p2p_lt_usleep()
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
                std::cout << "-----Latency test (w usleep)----" << std::endl;
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
                touch_buffers_lt(size);
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
                    
#if defined(TEST_LT_MPI_PROC_NULL) 
                    comm_kernel_lt_pnull_usleep(size);
#else
                    comm_kernel_lt_usleep(size);
#endif
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
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * sum_npairs) 
                        << std::endl;
                }
            }
            plat.clear();
        } 

        // Latency test using MPI Isend/Irecv, including work sum
        void p2p_lt_worksum()
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
                std::cout << "---------------------------------" << std::endl;
                std::cout << "-----Latency test (w worksum)----" << std::endl;
                std::cout << "---------------------------------" << std::endl;
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
                touch_buffers_lt(size);
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
                    
                    comm_kernel_lt_worksum(size);
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
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * sum_npairs) 
                        << std::endl;
                }
            }
            plat.clear();
        }
        
        // Latency test using MPI Isend/Irecv, including work max
        void p2p_lt_workmax()
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
                std::cout << "---------------------------------" << std::endl;
                std::cout << "-----Latency test (w workmax)----" << std::endl;
                std::cout << "---------------------------------" << std::endl;
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
                touch_buffers_lt(size);
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
                    
                    comm_kernel_lt_worksum(size);
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
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * sum_npairs) 
                        << std::endl;
                }
            }
            plat.clear();
        }

#ifndef SSTMAC
        // Latency test using all-to-all among graph neighbors 
        void nbr_ala_lt()
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
                std::cout << "---------------------------------" << std::endl;
                std::cout << "----Latency test (All-To-All)----" << std::endl;
                std::cout << "---------------------------------" << std::endl;
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
                touch_buffers_lt(size);
                MPI_Barrier(comm_);

                if (size > large_msg_size_) 
                {
                    loop = lt_loop_count_large_;
                    skip = lt_skip_count_large_;
		}
                
                // time communication kernel
                for (int l = 0; l < loop + skip; l++) 
                {           
                    if (l == skip)
                        t_start = MPI_Wtime();
                    
                    comm_kernel_lt_ala(size);
                }

                t_end = MPI_Wtime();
                t = (t_end - t_start) * 1.0e6 / (double)loop; 
                double t_sq = t*t;
                double sum_tsq = 0;
                
                // execution time stats
                MPI_Allreduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, comm_);
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

		double avg_t = sum_t / (double) size_;
                double avg_tsq = sum_tsq / (double) size_;
                double var = avg_tsq - (avg_t*avg_t);
                double stddev  = sqrt(var);
                
                double lmax = 0.0;
                MPI_Reduce(&t, &lmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
                MPI_Gather(&t, 1, MPI_DOUBLE, plat.data(), 1, MPI_DOUBLE, 0, comm_);
                
                if (rank_ == 0) 
                {
                    std::sort(plat.begin(), plat.end());
                    std::cout << std::setw(10) << size << std::setw(17) << avg_t
                        << std::setw(16) << lmax
                        << std::setw(16) << plat[n99-1]
                        << std::setw(16) << var
                        << std::setw(16) << stddev 
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * sum_npairs) 
                        << std::endl;
                }
            }
            plat.clear();
        } 
         
        // Latency test using all-gather among graph neighbors 
        void nbr_aga_lt()
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
                std::cout << "---------------------------------" << std::endl;
                std::cout << "----Latency test (All-Gather)----" << std::endl;
                std::cout << "---------------------------------" << std::endl;
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
                touch_buffers_lt(size);
                MPI_Barrier(comm_);

                if (size > large_msg_size_) 
                {
                    loop = lt_loop_count_large_;
                    skip = lt_skip_count_large_;
		}
                
                // time communication kernel
                for (int l = 0; l < loop + skip; l++) 
                {           
                    if (l == skip)
                        t_start = MPI_Wtime();
                    
                    comm_kernel_lt_aga(size);
                }

                t_end = MPI_Wtime();
                t = (t_end - t_start) * 1.0e6 / (double)loop; 
                double t_sq = t*t;
                double sum_tsq = 0;
                
                // execution time stats
                MPI_Allreduce(&t, &sum_t, 1, MPI_DOUBLE, MPI_SUM, comm_);
                MPI_Reduce(&t_sq, &sum_tsq, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

		double avg_t = sum_t / (double) size_;
                double avg_tsq = sum_tsq / (double) size_;
                double var = avg_tsq - (avg_t*avg_t);
                double stddev  = sqrt(var);
                
                double lmax = 0.0;
                MPI_Reduce(&t, &lmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
                MPI_Gather(&t, 1, MPI_DOUBLE, plat.data(), 1, MPI_DOUBLE, 0, comm_);
                
                if (rank_ == 0) 
                {
                    std::sort(plat.begin(), plat.end());
                    std::cout << std::setw(10) << size << std::setw(17) << avg_t
                        << std::setw(16) << lmax
                        << std::setw(16) << plat[n99-1]
                        << std::setw(16) << var
                        << std::setw(16) << stddev 
                        << std::setw(16) << stddev * ZCI / sqrt((double)loop * sum_npairs) 
                        << std::endl;
                }
            }
            plat.clear();
        }       
#endif

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
            
            // create new group/comm
            MPI_Group cur_grp, nbr_grp;
            MPI_Comm nbr_comm;

            MPI_Comm_group(comm_, &cur_grp);
            MPI_Group_incl(cur_grp, tgt_deg+1, exl_tgt.data(), &nbr_grp);
            MPI_Comm_create(comm_, nbr_grp, &nbr_comm);
            #ifndef SSTMAC 
            MPI_Group_rank(nbr_grp, &tgt_rank);
            MPI_Group_size(nbr_grp, &tgt_size);
            #else
            MPI_Comm_rank(nbr_comm, &tgt_rank);
            MPI_Comm_size(nbr_comm, &tgt_size);
            #endif
            
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
                            << std::setw(16) << stddev * ZCI / sqrt((double)avg_ng * loop) 
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

            #ifndef SSTMAC
            MPI_Group_rank(nbr_grp, &tgt_rank);
            MPI_Group_size(nbr_grp, &tgt_size);
            #else
            MPI_Comm_rank(nbr_comm, &tgt_rank);
            MPI_Comm_size(nbr_comm, &tgt_size);
            #endif

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
                    touch_buffers_lt(size);
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
                            << std::setw(16) << stddev * ZCI / sqrt((double)tgt_size * loop) 
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
        MPI_Comm comm_, nbr_comm_;
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
            comm_(MPI_COMM_NULL), rank_(MPI_PROC_NULL), size_(0), 
            ract_(0), sact_(nullptr), sctr_(nullptr),
            sbuf_(nullptr), rbuf_(nullptr), sreq_(nullptr), rreq_(MPI_REQUEST_NULL), 
            oldq_(nullptr), newq_(nullptr), nranks_done_(0), newq_count_(0), 
            oldq_count_(0), seed_(DEF_BFS_SEED), edge_visit_count_(0)
        {
            const GraphElem lnv = g_->get_lnv();
            comm_ = g_->get_comm();

            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            visited_ = new GraphElem[lnv];
            pred_    = new GraphElem[lnv];
            oldq_    = new GraphElem[lnv];
            newq_    = new GraphElem[lnv];
            rbuf_    = new GraphElem[bufsize_*2];
            sbuf_    = new GraphElem[size_*bufsize_*2];
            sctr_    = new GraphElem[size_];
            sreq_    = new MPI_Request[size_];
            sact_    = new GraphElem[size_];

            std::fill(sreq_, sreq_ + size_, MPI_REQUEST_NULL);
            std::fill(sctr_, sctr_ + size_, 0);
            std::fill(sact_, sact_ + size_, 0);
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
            while (ract_) 
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
                        for (GraphElem j = 0; j < count; j += 2) 
                        {
                            GraphElem tgt = rbuf_[j];
                            GraphElem src = rbuf_[j + 1];

                            /* Process one incoming edge. */
                            assert (g_->get_owner(tgt) == rank_);
                            if (!test_visited(tgt)) 
                            {
                                set_visited(tgt);
                                pred_[g_->global_to_local(tgt)] = src;
                                newq_[newq_count_++] = tgt;
                                edge_visit_count_++;
                            }
                        }
                    }

                    /* Restart the receive if more messages will be coming. */
                    if (nranks_done_ < size_) 
                    {
                        MPI_Irecv(rbuf_, bufsize_ * 2, MPI_GRAPH_TYPE, MPI_ANY_SOURCE, 0, comm_, &rreq_);
                        ract_ = 1;
                    }
                } 
                else 
                    break;
            }

            /* Mark any sends that completed as inactive so their buffers can be reused. */
            for (int c = 0; c < size_; ++c) 
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
            MPI_Isend(&sbuf_[owner * bufsize_ * 2], bufsize_ * 2, MPI_GRAPH_TYPE, 
                    owner, 0, comm_, &sreq_[owner]);

            sact_[owner] = 1;
            sctr_[owner] = 0;
        }
        
        void nbsend_count(GraphElem owner)
        {
            MPI_Isend(&sbuf_[owner * bufsize_ * 2], sctr_[owner], MPI_GRAPH_TYPE, 
                    owner, 0, comm_, &sreq_[owner]);

            sact_[owner] = 1;
            sctr_[owner] = 0;
        }

        void nbsend_zero(GraphElem owner)
        {
            /* Base address is meaningless for 0-sends. */
            MPI_Isend(&sbuf_[0], 0, MPI_GRAPH_TYPE, owner, 0, comm_, &sreq_[owner]);

            sact_[owner] = 1;
        }

        // reimplementation of graph500 BFS
        void run_bfs(GraphElem root) 
        {
#if defined(USE_ALLREDUCE_FOR_EXIT)
            GraphElem global_newq_count;
#else      
            bool done = false, nbar_active = false; 
            MPI_Request nbar_req = MPI_REQUEST_NULL;
#endif
            /* Mark the root and put it into the queue. */
            if (g_->get_owner(root) == rank_) 
            {
                set_visited(root);
                pred_[g_->global_to_local(root)] = root;
                oldq_[oldq_count_++] = root;
                edge_visit_count_++;
            }

            process_msgs();

#if defined(USE_ALLREDUCE_FOR_EXIT)
                while(1)
#else
                while(!done)
#endif
                {
                    memset(sctr_, 0, size_ * sizeof(GraphElem));
                    nranks_done_ = 0;

                    /* Start the initial receive. */
                    if (nranks_done_ < size_) 
                    {
                        MPI_Irecv(rbuf_, bufsize_ * 2, MPI_GRAPH_TYPE, MPI_ANY_SOURCE, 0, comm_, &rreq_);
                        ract_ = 1;
                    }

                    /* Step through the current level's queue. */
                    for (GraphElem i = 0; i < oldq_count_; ++i) 
                    {
                        process_msgs();

                        assert (g_->get_owner(oldq_[i]) == rank_);
                        assert (pred_[g_->global_to_local(oldq_[i])] >= 0 && pred_[g_->global_to_local(oldq_[i])] < g_->get_nv());
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

                            if (owner == rank_)
                            {
                                if (!test_visited(edge.tail_)) 
                                {
                                    set_visited(edge.tail_);
                                    pred_[g_->global_to_local(edge.tail_)] = src;
                                    newq_[newq_count_++] = edge.tail_;
                                    edge_visit_count_++;
                                }
                            }
                            else
                            {
                                /* Wait for buffer to be available */
                                while (sact_[owner]) 
                                    process_msgs();

                                GraphElem c = sctr_[owner];
                                sbuf_[owner * bufsize_ * 2 + c]     = edge.tail_;
                                sbuf_[owner * bufsize_ * 2 + c + 1] = src;
                                sctr_[owner] += 2;

                                if (sctr_[owner] == (bufsize_ * 2))
                                    nbsend(owner);
                            }
                        }
                    }

                    /* Flush any coalescing buffers that still have messages. */
                    for (int p = 0; p < size_; p++)
                    {
                        if (sctr_[p] != 0) 
                        {
                            while (sact_[p]) 
                                process_msgs();

                            nbsend_count(p);
                        }

                        /* Wait until all sends to this destination are done. */
                        while (sact_[p]) 
                            process_msgs();

                        /* Tell the destination that we are done sending to them. */
                        /* Signal no more sends */
                        nbsend_zero(p);

                        while (sact_[p]) 
                            process_msgs();
                    }

                    /* Wait until everyone else is done (and thus couldn't send us any more
                     * messages). */
                    while (nranks_done_ < size_) 
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
                    newq_ = tmp;

                    oldq_count_ = newq_count_;
                    newq_count_ = 0;
                }
        }

        void run_test(GraphElem nbfs_roots=DEF_BFS_ROOTS)
        {
            std::seed_seq seed{seed_};
            std::mt19937 gen{seed};
            std::uniform_int_distribution<GraphElem> uid(0, g_->get_nv()-1); 

            std::vector<GraphElem> bfs_roots(nbfs_roots);
            std::vector<double> bfs_times(nbfs_roots);

            double t1 = MPI_Wtime();

            // calculate bfs roots
            GraphElem counter = 0, bfs_root_idx = 0, nv = g_->get_nv();
            
            for (bfs_root_idx = 0; bfs_root_idx < nbfs_roots; ++bfs_root_idx) 
            {
              GraphElem root;

              while (1) 
              {
                root = uid(gen);

                counter++;
                
                if (counter > nv) 
                  break;

                int is_duplicate = 0;

                for (GraphElem i = 0; i < bfs_root_idx; ++i) 
                {
                  if (root == bfs_roots[i]) 
                  {
                    is_duplicate = 1;
                    break;
                  }
                }

                if (is_duplicate) 
                  continue;

                int root_ok = 0;
                if (g_->get_owner(root) == rank_)
                {
                  GraphElem e0, e1;
                  g_->edge_range(g_->global_to_local(root), e0, e1);
                  if ((e0 + 1) != e1)
                    root_ok = 1;
                }
                
                MPI_Barrier(comm_);
                MPI_Allreduce(MPI_IN_PLACE, &root_ok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                
                if (root_ok) 
                  break;  
              }

              bfs_roots[bfs_root_idx] = root;
            }

            nbfs_roots = bfs_root_idx;

            double t2 = MPI_Wtime() - t1;
            double root_t = 0.0;
            MPI_Reduce(&t2, &root_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
            if (rank_ == 0)
              fprintf(stderr, "Average time(s) taken to calculate %d BFS root vertices: %f\n", nbfs_roots, root_t);

            int test_ctr = 0;

            for (GraphElem const& r : bfs_roots)
            {
                if (rank_ == 0) 
                    fprintf(stderr, "Running BFS %d\n", test_ctr);
            
                /* Set all vertices to "not visited." */
                std::fill(pred_, pred_ + g_->get_lnv(), 0);

                /* Do the actual BFS. */
                double bfs_start = MPI_Wtime(), g_bfs_time = 0.0;
                run_bfs(r);
                double bfs_stop = MPI_Wtime();
                bfs_times[test_ctr] = bfs_stop - bfs_start;
                MPI_Allreduce(&bfs_times[test_ctr], &g_bfs_time, 1, MPI_DOUBLE, MPI_SUM, comm_);

                if (rank_ == 0)
                {
                    double avgt = ((double)(g_bfs_time / (double)size_));
                    bfs_times[test_ctr] = avgt;
                    fprintf(stderr, "Average time(s), TEPS for BFS %d, %d: %f\n", test_ctr, r, avgt);
                }
                    
                test_ctr++;
            }
            
            GraphElem ecg = 0; /* Total edge visitations. */
            MPI_Reduce(&edge_visit_count_, &ecg, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
                
            if (rank_ == 0)
            {
                double avgt_nroots = std::accumulate(bfs_times.begin(), bfs_times.end(), 0.0) / bfs_roots.size();
                fprintf(stderr, "-------------------------------------\n");
                fprintf(stderr, "Average time(s), TEPS across %d roots: %f, %g\n", bfs_roots.size(), avgt_nroots, (double)((double)ecg / avgt_nroots));
                fprintf(stderr, "-------------------------------------\n");
            }
        }

        GraphWeight get_weight(GraphElem u, GraphElem v)
        {
            GraphElem e0, e1;
            g_->edge_range(g_->global_to_local(u), e0, e1);
            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                if (edge.tail_ == v)
                    return edge.tail_;
            }
            return 0;
        }

        void nbsend()
        {
            for (GraphElem owner = 0; owner < size_; owner++)
            {
                if (sctr_[owner] > 0 && !sact_[owner])
                {
                    MPI_Isend(&sbuf_[owner * bufsize_ * 2], bufsize_ * 2, MPI_GRAPH_TYPE, 
                            owner, 0, comm_, &sreq_[owner]);

                    sact_[owner] = 1;
                    sctr_[owner] = 0;
                }
            }
        }

        void run_test_sssp(GraphElem nsssp_roots=DEF_BFS_ROOTS)
        {
            std::seed_seq seed{seed_};
            std::mt19937 gen{seed};
            std::uniform_int_distribution<GraphElem> uid(0, g_->get_nv()-1); 

            std::vector<GraphElem> sssp_roots(nsssp_roots);
            std::vector<double> sssp_times(nsssp_roots);

            double t1 = MPI_Wtime();

            // calculate bfs roots
            GraphElem counter = 0, sssp_root_idx = 0, nv = g_->get_nv();
            
            for (sssp_root_idx = 0; sssp_root_idx < nsssp_roots; ++sssp_root_idx) 
            {
              GraphElem root;

              while (1) 
              {
                root = uid(gen);

                counter++;
                
                if (counter > nv) 
                  break;
                
                int is_duplicate = 0;

                for (GraphElem i = 0; i < sssp_root_idx; ++i) 
                {
                  if (root == sssp_roots[i]) 
                  {
                    is_duplicate = 1;
                    break;
                  }
                }

                if (is_duplicate) 
                  continue;

                int root_ok = 0;
                if (g_->get_owner(root) == rank_)
                {
                  GraphElem e0, e1;
                  g_->edge_range(g_->global_to_local(root), e0, e1);
                  if ((e0 + 1) != e1)
                    root_ok = 1;
                }
                
                MPI_Barrier(comm_);
                MPI_Allreduce(MPI_IN_PLACE, &root_ok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                
                if (root_ok) 
                  break;
              }

              sssp_roots[sssp_root_idx] = root;
            }

            nsssp_roots = sssp_root_idx;

            double t2 = MPI_Wtime() - t1;
            double root_t = 0.0;
            MPI_Reduce(&t2, &root_t, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
            if (rank_ == 0)
              fprintf(stderr, "Average time(s) taken to calculate %ld SSSP root vertices: %f\n", nsssp_roots, root_t);

            int test_ctr = 0;

            for (GraphElem const& r : sssp_roots)
            {
                if (rank_ == 0) 
                    fprintf(stderr, "Running SSSP %d\n", test_ctr);
            
                /* Set all vertices to "not visited." */
                std::fill(pred_, pred_ + g_->get_lnv(), 0);
                std::fill(dist_, dist_ + g_->get_lnv(), -1);

                /* Do the actual SSSP. */
                double sssp_start = MPI_Wtime(), g_sssp_time = 0.0;
                run_sssp(r);
                double sssp_stop = MPI_Wtime();
                sssp_times[test_ctr] = sssp_stop - sssp_start;
                MPI_Allreduce(&sssp_times[test_ctr], &g_sssp_time, 1, MPI_DOUBLE, MPI_SUM, comm_);

                if (rank_ == 0)
                {
                    double avgt = ((double)(g_sssp_time / (double)size_));
                    sssp_times[test_ctr] = avgt;
                    fprintf(stderr, "Average time(s), TEPS for SSSP %d, %d: %f\n", test_ctr, r, avgt);
                }
                    
                test_ctr++;
            }
            
            GraphElem ecg = 0; /* Total edge visitations. */
            MPI_Reduce(&edge_visit_count_, &ecg, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
                
            if (rank_ == 0)
            {
                double avgt_nroots = std::accumulate(sssp_times.begin(), sssp_times.end(), 0.0) / sssp_roots.size();
                fprintf(stderr, "-------------------------------------\n");
                fprintf(stderr, "Average time(s), TEPS across %d roots: %f, %g\n", sssp_roots.size(), avgt_nroots, (double)((double)ecg / avgt_nroots));
                fprintf(stderr, "-------------------------------------\n");
            }
        }

        void run_sssp(GraphElem root) 
        {
            GraphElem sum = 0, qc = 0, q2c = 0;
            GraphWeight delta = 10.0, mindelta = 0.0, maxdelta = delta;

            const GraphElem lnv = g_->get_lnv();

            GraphElem* q1  = new GraphElem[lnv];
            GraphElem* q2  = new GraphElem[lnv];

            std::fill(q1, q1 + lnv, -1);
            std::fill(q2, q2 + lnv, -1);

            MPI_Datatype edgeType;
            createEdgeTupleType(&edgeType);

            if (g_->get_owner(root) == rank_) 
            {
                q1[0] = g_->global_to_local(root); 
                qc = 1;
                pred_[g_->global_to_local(root)] = root;
                dist_[g_->global_to_local(root)] = 0.0;
                set_visited(root);
            }

            sum=1;

            while(sum != 0) 
            {
#ifdef PRINT_DEBUG
                double t0 = MPI_Wtime();
#endif
                std::vector<std::vector<EdgeTuple>> buf(size_, std::vector<EdgeTuple>());
                std::vector<EdgeTuple> sbuf, rbuf;
                std::vector<int> scounts(size_, 0), sdispls(size_, 0), rcounts(size_, 0), rdispls(size_, 0);
                GraphElem sdisp = 0, rdisp = 0;

                //1. iterate over light edges
                while (sum!=0) 
                {
                    for(GraphElem i = 0; i < qc; i++)
                    {
                        GraphElem e0, e1;
                        g_->edge_range(q1[i], e0, e1);
                        for (GraphElem e = e0; e < e1; e++)
                        {
                            Edge const& edge = g_->get_edge(e);
                            if (edge.tail_ < delta)
                            {
                                const int owner = g_->get_owner(edge.tail_);
                                buf[owner].push_back({q1[i], edge.tail_, (dist_[g_->global_to_local(q1[i])] + edge.weight_)});
                            }
                        }
                    }

                    for (GraphElem p = 0; p < size_; p++)
                    {
                        std::copy(buf[p].begin(), buf[p].end(), std::back_inserter(sbuf));
                        sdispls[p] = sdisp;
                        scounts[p] = buf[p].size();
                        sdisp += scounts[p];
                        buf[p].clear();
                    }

                    buf.clear();

                    MPI_Alltoall(scounts.data(), 1, MPI_GRAPH_TYPE, rcounts.data(), 1, MPI_GRAPH_TYPE, comm_);

                    for (GraphElem p = 0; p < size_; p++)
                    {
                        rdispls[p] = rdisp;
                        rdisp += rcounts[p];
                    }

                    rbuf.resize(rdisp);

                    MPI_Alltoallv(sbuf.data(), scounts.data(), sdispls.data(), edgeType, rbuf.data(), rcounts.data(),
                            rdispls.data(), edgeType, comm_);

                    for (GraphElem j = 0; j < rdisp; j++) 
                    {
                        EdgeTuple tup = rbuf[j];
                        GraphElem const& u = tup.ij_[0];
                        GraphElem const& v = tup.ij_[1];
                        GraphWeight const& dtv = tup.w_;

                        if (dist_[g_->global_to_local(v)] != -1 && dist_[g_->global_to_local(v)] > dtv)
                        {
                            if (!test_visited(v)) 
                            {
                                dist_[g_->global_to_local(v)] = dtv;
                                q2[q2c++] = v;
                                pred_[g_->global_to_local(v)] = u;
                                set_visited(v);
                                edge_visit_count_++;
                            }
                        }
                    }

                    qc = q2c;
                    q2c = 0;
                    std::swap(q1, q2);
                    sum = qc;

                    MPI_Allreduce(&qc, &sum, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);

                    sbuf.clear();
                    rbuf.clear();
                    scounts.clear();
                    sdispls.clear();
                    rcounts.clear();
                    rdispls.clear();
                    sdisp = 0;
                    rdisp = 0;
                }

                MPI_Barrier(comm_);

                //2. iterate over S and heavy edges
                for (GraphElem i = 0; i < lnv; i++)
                {
                    if (dist_[i] >= mindelta && dist_[i] < maxdelta)
                    {
                        GraphElem e0, e1;
                        g_->edge_range(i, e0, e1);
                        for (GraphElem e = e0; e < e1; e++)
                        {
                            Edge const& edge = g_->get_edge(e);
                            if (edge.weight_ >= delta)
                            {
                                const int owner = g_->get_owner(edge.tail_);
                                buf[owner].push_back({g_->local_to_global(i), edge.tail_, (dist_[i] + edge.weight_)});
                            }
                        }
                    }
                }

                for (GraphElem p = 0; p < size_; p++)
                {
                    std::copy(buf[p].begin(), buf[p].end(), std::back_inserter(sbuf));
                    sdispls[p] = sdisp;
                    scounts[p] = buf[p].size();
                    sdisp += scounts[p];
                    buf[p].clear();
                }

                buf.clear();

                MPI_Alltoall(scounts.data(), 1, MPI_GRAPH_TYPE, rcounts.data(), 1, MPI_GRAPH_TYPE, comm_);

                for (GraphElem p = 0; p < size_; p++)
                {
                    rdispls[p] = rdisp;
                    rdisp += rcounts[p];
                }

                rbuf.resize(rdisp);

                MPI_Alltoallv(sbuf.data(), scounts.data(), sdispls.data(), edgeType, rbuf.data(), rcounts.data(),
                        rdispls.data(), edgeType, comm_);

                for (GraphElem j = 0; j < rdisp; j++) 
                {
                    EdgeTuple tup = rbuf[j];
                    GraphElem const& u = tup.ij_[0];
                    GraphElem const& v = tup.ij_[1];
                    GraphWeight const& dtv = tup.w_;

                    if (dist_[g_->global_to_local(v)] != -1 && dist_[g_->global_to_local(v)] > dtv)
                    {
                        if (!test_visited(v)) 
                        {
                            dist_[g_->global_to_local(v)] = dtv;
                            pred_[g_->global_to_local(v)] = u;
                            set_visited(v);
                            edge_visit_count_++;
                        }
                    }
                }


                mindelta = maxdelta;
                maxdelta += delta;
                qc = 0;
                sum = 0;

                //3. Bucket processing and checking termination condition
                for (GraphElem i = 0; i < lnv; i++)
                {
                    if (dist_[i] >= mindelta)
                    {
                        sum++; //how many are still to be processed
                        if (dist_[i] < maxdelta)
                            q1[qc++] = i; //this is lowest bucket
                    }
                }

                GraphElem lsum = sum;
                MPI_Allreduce(&lsum, &sum, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);

#ifdef PRINT_DEBUG
                t0 -= MPI_Wtime();
                if (rank_ == 0) 
                    std::cout << "min/max delta: " << mindelta << ", " << maxdelta << std::endl;
#endif

            }
            
            freeEdgeTupleType(&edgeType);
            delete []q1;
            delete []q2;
        }

    private:
        Graph* g_;

        int rank_, size_;
        MPI_Comm comm_;

        GraphElem bufsize_, newq_count_, oldq_count_, nranks_done_, ract_, seed_, edge_visit_count_;
        GraphElem *sbuf_, *rbuf_, *pred_, *visited_, *oldq_, *newq_, *sctr_, *sact_;
        GraphWeight *dist_;

        MPI_Request *sreq_, rreq_;
};

#endif
