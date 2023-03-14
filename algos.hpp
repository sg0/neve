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

#endif
