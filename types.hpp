#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <random>
#include <utility>
#include <cstring>

#ifdef USE_32_BIT_GRAPH
using GraphElem = int32_t;
using GraphWeight = float;
#ifdef USE_CUDA
using GraphElem2 = int2;
using GraphWeight2 = float2;
using Float2 = float2;
#endif
#if defined(USE_SHARED_MEMORY)
typedef std::aligned_storage<sizeof(GraphElem),alignof(GraphElem)>::type __GraphElem__;
typedef std::aligned_storage<sizeof(GraphWeight),alignof(GraphWeight)>::type __GraphWeight__;
#else
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT32_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_FLOAT;
#endif
#else
using GraphElem = int64_t;
using GraphWeight = double;
#ifdef USE_CUDA
using GraphElem2 = longlong2;
using GraphWeight2 = double2;
using Float2 = double2;
#endif
#if defined(USE_SHARED_MEMORY)
typedef std::aligned_storage<sizeof(GraphElem),alignof(GraphElem)>::type __GraphElem__;
typedef std::aligned_storage<sizeof(GraphWeight),alignof(GraphWeight)>::type __GraphWeight__;
#else
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT64_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_DOUBLE;
#endif
#endif

#ifdef EDGE_AS_VERTEX_PAIR
struct Edge
{   
    GraphElem head_, tail_;
    GraphWeight weight_;
    
    Edge(): head_(-1), tail_(-1), weight_(0.0) {}
};
#else
struct Edge
{   
    GraphElem tail_;
    GraphWeight weight_;
    
    Edge(): tail_(-1), weight_(0.0) {}
};
#endif
#endif

