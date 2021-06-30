#pragma once
#ifndef GRAPH_CUH_
#define GRAPH_CUH_

#include "utils.hpp"
#include <cooperative_groups.h> 
namespace cg = cooperative_groups;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,__double_as_longlong(__longlong_as_double(assumed)+val));
    } while(assumed != old);

    return __longlong_as_double(old);
}
#endif

template<const int blocksize>
__global__
void nbrscan_kernel(GraphWeight* __restrict__ edge_weights, Edge* 
__restrict__ edge_list, GraphElem* __restrict__ edge_indices,GraphElem nv)
{
    __shared__ GraphElem range[2];
    GraphElem step = gridDim.x;
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
       if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
	__syncthreads();
        
        GraphElem start = range[0];
        GraphElem end = range[1];
        //GraphElem start = edge_indices[i];
        //GraphElem end = edge_indices[i+1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
            edge_weights[e] = edge_list[e].weight_;
        __syncthreads();
    }
    //if(threadIdx.x == 0 and blockIdx.x == 0) 
    //printf("%lf\n", (double)edge_weights[0]);
} 

template<const int blocksize>
__global__
void nbrsum_kernel(GraphWeight* __restrict__ vertex_degree, Edge* __restrict__ edge_list, 
GraphElem* __restrict__ edge_indices, GraphElem nv)
{
    __shared__ GraphElem range[2];
    volatile  __shared__ GraphWeight data[blocksize];
    GraphElem step = gridDim.x;
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
        GraphWeight w = 0.;
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
        __syncthreads();

        GraphElem start = range[0];
        GraphElem end = range[1];

        //GraphElem start = edge_indices[i];
        //GraphElem end = edge_indices[i+1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
            w += edge_list[e].weight_;
        data[threadIdx.x] = w;
        __syncthreads();
        for (unsigned int s=blocksize/2; s>=32; s>>=1) 
        {
            if (threadIdx.x < s)
                data[threadIdx.x] += data[threadIdx.x + s];
            __syncthreads();
        }
        w = data[threadIdx.x%32];
        //__syncthreads();
        for (int offset = 16; offset > 0; offset /= 2)
            w += __shfl_down_sync(0xffffffff, w, offset);

        if(threadIdx.x == 0)
            vertex_degree[i] += w;
        __syncthreads();
    }
}

template<const int blocksize>
__global__
void nbrmax_kernel(GraphWeight* __restrict__ vertex_degree, Edge* __restrict__ edge_list, 
GraphElem* __restrict__ edge_indices, GraphElem nv)
{
    __shared__ GraphElem range[2];
    volatile __shared__ GraphWeight data [blocksize];
    GraphElem step = gridDim.x;
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
        GraphWeight max = 0.;
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
        __syncthreads();

        GraphElem start = range[0];
        GraphElem end = range[1];

        //GraphElem start = edge_indices[i];
        //GraphElem end = edge_indices[i+1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
        {
            GraphWeight w = edge_list[e].weight_;
            if(max < w)
                max = w;
        }
        data[threadIdx.x] = max;
        __syncthreads();
        for (unsigned int s=blocksize/2; s>=32; s>>=1)
        {
            if (threadIdx.x < s && data[threadIdx.x+s] > data[threadIdx.x])
                data[threadIdx.x] = data[threadIdx.x+s];
            __syncthreads();
        }
        max = data[threadIdx.x%32];
        //__syncthreads();
        for (int offset = 16; offset > 0; offset /= 2)
        {
            GraphWeight tmp = __shfl_down_sync(0xffffffff, max, offset);
            max = (tmp > max) ? tmp : max;
        }

        if(threadIdx.x == 0)
            vertex_degree[i] = max; 
        __syncthreads();
    } 
}
#if 0
template<const Int BlockSize>
nbr_max_order_reduce_kernel
(
    GraphElem* __restrict__ indices, 
    GraphElem* __restrict__ orders, 
    GraphElem numVertices
)
{
    __shared__ GraphElem max_shared[BlockSize];

    max_shared[threadIdx.x] = 0;

    GraphElem u0 = threadIdx.x + BlockSize * blockIdx.x; 

    for(GraphElem i = u0; i < numVertices; i += BlockSize*gridDim.x)
    {    
        GraphElem order = indices[i+1]-indices[i];
        if(max_shared[threadIdx.x] < order) 
            max_shared[threadIdx.x] = order;
    }
    __syncthreads();

    for (unsigned int s = BlockSize/2; s >= 32; s>>=1)
    {
        if (threadIdx.x < s && max_shared[threadIdx.x+s] > max_shared[threadIdx.x])
            max_shared[threadIdx.x] = max_shared[threadIdx.x+s];
        __syncthreads();
    }

    GraphElem max = max_shared[threadIdx.x%32];
    for (int offset = 16; offset > 0; offset /= 2)
    {
        GraphElem tmp = __shfl_down_sync(0xffffffff, max, offset);
        max = (tmp > max) ? tmp : max;
    }

    if(threadIdx.x == 0)
        orders[blockIdx.x] = max; 
}

template<const Int BlockSize>
nbr_max_order_kernel
(
    GraphElem* __restrict__ orders, 
    GraphElem numVertices
)
{
    __shared__ GraphElem max_shared[BlockSize];

    max_shared[threadIdx.x] = 0;

    GraphElem u0 = threadIdx.x;

    for(GraphElem i = u0; i < numVertices; i += BlockSize)
    {
        GraphElem order = orders[i];
        if(max_shared[threadIdx.x] < order) 
            max_shared[threadIdx.x] = order;
    }
    __syncthreads();

    for (unsigned int s = BlockSize/2; s >= 32; s>>=1)
    {
        if (threadIdx.x < s && max_shared[threadIdx.x+s] > max_shared[threadIdx.x])
            max_shared[threadIdx.x] = max_shared[threadIdx.x+s];
        __syncthreads();
    }

    GraphElem max = max_shared[threadIdx.x%32];
    for (int offset = 16; offset > 0; offset /= 2)
    {
        GraphElem tmp = __shfl_down_sync(0xffffffff, max, offset);
        max = (tmp > max) ? tmp : max;
    }

    if(threadIdx.x == 0)
        orders[0] = max;

}

template<const int WarpSize, const int BlockSize>
__global__
void fill_index_orders_kernel
(
    GraphElem* __restrict__ index_orders,
    GraphElem* __restrict__ indices,
    GraphElem numVertices 
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();
    #if __CUDA_ARCH__ > 700
    cg::thread_group warp = cg::partition<WarpSize>(block);
    #else 
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);
    #endif
    const unsigned block_thread_id = block.thread_rank();
    const unsigned warp_thread_id = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_thread_id/WarpSize)*2];

    GraphElem u0 = block_thread_id/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < numVertices; u += (GraphElem)(BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_thread_id == 0)
        {
            t_ranges[0] = indices[u+0];
            t_ranges[1] = indices[u+1];
        }
        warp.sync(); 
        GraphElem start0 = t_ranges[0];               
        GraphElem start = start0 + warp_thread_id;
        GraphElem end = t_ranges[1];
        for(GraphElem i = start; i < end; i += WarpSize)
             index_orders[i] = i-start0;
        warp.sync();
    } 
}

template<const int BlockSize>
__global__
void fill_edges_community_ids_kernel
(
    GraphElem* __restrict__ edges,
    GraphElem* __restrict__ commIds,
    GraphElem numEdges
)
{
    GraphElem start0 = threadIdx.x + blockDim.x*BlockSize;
    for(GraphElem i = start0; i < numEdges; i += BlockSize*gridDim.x)
        edges[i] = commIds[edges[i]];
}

template<const int WarpSize, const int BlockSize>
__global__
void reorder_edges_by_keys_kernel
(
    GraphElem* __restrict__ edges,
    GraphElem* __restrict__ index_orders,
    GraphElem* __restrict__ indices,
    GraphElem numVertices
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();

    #if __CUDA_ARCH__ > 700
    cg::thread_group warp = cg::partition<WarpSize>(block);
    #else
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);
    #endif

    const unsigned block_thread_id = block.thread_rank();
    const unsigned warp_thread_id = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_thread_id/WarpSize)*2];

    GraphElem u0 = block_thread_id/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < numVertices; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_thread_id == 0)
        {
            t_ranges[0] = indices[u+0];
            t_ranges[1] = indices[u+1];
        }
        warp.sync(); 
        GraphElem start = t_ranges[0] + warp_thread_id;
        GraphElem end = t_ranges[1];
        for(GraphElem i = start; i < end; i += WarpSize)
        {
            GraphElem pos = index_orders[i];
            index_orders[i] = edges[pos];
        }
        warp.sync();
        for(GraphElem i = start; i < end; i += WarpSize)
            edges[i] = index_orders[i];
        warp.sync();
    } 
}

#ifdef USE_32_BIT_GRAPH
template<const int WarpSize, const int BlockSize>
__global__
void reorder_weights_by_keys_kernel
(
    float*   __restrict__ weights,
    int32_t* __restrict__ index_orders,
    int32_t* __restrict__ indices,
    GraphElem numVertices
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    thread_block block = cg::this_thread_block();
    #if __CUDA_ARCH__ > 700
    cg::thread_group warp = cg::partition<WarpSize>(block);
    #else
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);
    #endif
    const unsigned block_thread_id = block.thread_rank();
    const unsigned warp_thread_id = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_thread_id/WarpSize)*2];

    GraphElem u0 = block_thread_id/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < numVertices; u += BlockSize/WarpSize*gridDim.x)
    {
        if(warp_thread_id == 0)
        {
            t_ranges[0] = indices[u+0];
            t_ranges[1] = indices[u+1];
        }
        warp.sync(); 
        GraphElem start = t_ranges[0] + warp_thread_id;
        GraphElem end = t_ranges[1];
        for(GraphElem i = start; i < end; i += WarpSize)
        {
            uint32_t pos = index_orders[i];
            index_orders[i] = __float_as_uint(weights[pos]);
        }
        warp.sync();
        for(GraphElem i = start; i < end; i += WarpSize)
            weights[i] = __uint_as_float(index_orders[i]);
        warp.sync();
    } 
}

#else

template<const int WarpSize, const int BlockSize>
__global__
void reorder_weights_by_keys_kernel
(
    double*  __restrict__ weights,
    int64_t* __restrict__ index_orders,
    int64_t* __restrict__ indices,
    GraphElem numVertices
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();

    #if __CUDA_ARCH__ > 700
    cg::thread_group warp = cg::partition<WarpSize>(block);
    #else
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);
    #endif

    const unsigned block_thread_id = block.thread_rank();
    const unsigned warp_thread_id = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_thread_id/WarpSize)*2];

    GraphElem u0 = block_thread_id/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < numVertices; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_thread_id == 0)
        {
            t_ranges[0] = indices[u+0];
            t_ranges[1] = indices[u+1];
        }
        warp.sync(); 
        GraphElem start = t_ranges[0] + warp_thread_id;
        GraphElem end = t_ranges[1];
        for(GraphElem i = start; i < end; i += WarpSize)
        {
            long long pos = index_orders[i];
            index_orders[i] =  __double_as_longlong(weights[pos]);
        }
        warp.sync();
        for(GraphElem i = start; i < end; i += WarpSize)
            weights[i] = __longlong_as_double(index_orders[i]);
        warp.sync();
    } 
}

#endif

template<const int BlockSize>
__global__
void compute_community_weighted_orders_kernel
(
    GraphWeight* __restrict__ commWeights,
    GraphWeight* __restrict__ weighted_orders,
    GraphElem*   __restrict__ commIds,
    GraphElem numVertices
)
{
    GraphElem u0 = threadIdx.x+BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < numVertices; i += BlockSize*gridDim.x)
    {
        GraphElem comm_id = commIds[i];
        GraphWeight w = weighted_orders[i];
        atomicAdd(commWeights+comm_id, w);
    }
}
#endif
#endif
