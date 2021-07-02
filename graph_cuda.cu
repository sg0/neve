#include "types.hpp"
#include "cuda_wrapper.hpp"
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

template<const int WarpSize, const int BlockSize>
__global__
void reorder_weights_by_keys_kernel
(
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   __restrict__ indexOrders,
    GraphElem*   __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid  = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*2];

    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            #ifdef USE_32BIT_GRAPH
            indexOrders[i] = __float_as_int(edgeWeights[pos+start]);
            #else
            indexOrders[i] = __double_as_longlong(edgeWeights[pos+start]);
            #endif
        }
        warp.sync();
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
            #ifdef USE_32BIT_GRAPH
            edgeWeights[i] = __uint_as_float(indexOrders[i]);
            #else
            edgeWeights[i] = __longlong_as_double(indexOrders[i]);
            #endif
        warp.sync();
    } 
}


void reorder_weights_by_keys_cuda
( 
    GraphWeight* edgeWeights, 
    GraphElem* indexOrders, 
    GraphElem* indices, 
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    GraphElem nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE); 
    reorder_weights_by_keys_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01,0,stream>>>(edgeWeights, indexOrders, indices, v0, e0, nv); 
}

template<const int WarpSize, const int BlockSize>
__global__
void reorder_edges_by_keys_kernel
(
    GraphElem* __restrict__ edges,
    GraphElem* __restrict__ indexOrders,
    GraphElem* __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid  = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*2];

    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end = t_ranges[1]-e_base;
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            indexOrders[i] = edges[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
            edges[i] = indexOrders[i];
        warp.sync();
    } 
}

void reorder_edges_by_keys_cuda
(
    GraphElem* edges, 
    GraphElem* indexOrders, 
    GraphElem* indices, 
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE);
    reorder_edges_by_keys_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(edges, indexOrders, indices, v0, e0, nv);
}

template<const int WarpSize, const int BlockSize>
__global__
void fill_edges_community_ids_kernel
(
    GraphElem2* __restrict__ commIdKeys,
    GraphElem*  __restrict__ edges,
    GraphElem*  __restrict__ indices,
    GraphElem*  __restrict__ commIds,
    const GraphElem v_base, 
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[BlockSize/WarpSize*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*2];

    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;

    for(GraphElem u = u0; u < nv; u += (GraphElem)BlockSize/WarpSize*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();
 
        GraphElem start = t_ranges[0] + warp_tid-e_base;               
        GraphElem end = t_ranges[1]-e_base;
        for(GraphElem i = start; i < end; i += WarpSize)
        {
            GraphElem commId = commIds[edges[i]];    
            #ifdef USE_32BIT_GRAPH
            commIdKeys[i] = make_int2(u, commId);
            #else
            commIdKeys[i] = make_longlong2(u, commId);
            #endif
        }
        warp.sync();
    } 
}

void fill_edges_community_ids_cuda
(
    GraphElem2* commIdKeys, 
    GraphElem* edges,
    GraphElem* indices,
    GraphElem* commIds,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE);    
    fill_edges_community_ids_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
    (commIdKeys, edges, indices, commIds, v0, e0, nv);
}

template<const int WarpSize, const int BlockSize>
__global__
void fill_index_orders_kernel
(
    GraphElem* __restrict__ indexOrders,
    GraphElem* __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv 
)
{
    __shared__ GraphElem ranges[BlockSize/WarpSize*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*2];

    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;

    for(GraphElem u = u0; u < nv; u += (GraphElem)BlockSize/WarpSize*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();
 
        GraphElem start = t_ranges[0];               
        GraphElem end = t_ranges[1];
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
             indexOrders[i-e_base] = i-start;
        warp.sync();
    } 
}

void fill_index_orders_cuda
(
    GraphElem* indexOrders,
    GraphElem* indices,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    int nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE);    
    fill_index_orders_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
    (indexOrders, indices, v0, e0, nv);
}

#if 0
template<const int WarpSize, const int blocksize>
__global__
void sum_vertex_weights_kernel
(GraphWeight* __restrict__ vertex_degree, GraphWeight* __restrict__ weights,
GraphElem* __restrict__ edge_indices, const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem range[2];
    volatile  __shared__ GraphWeight data[blocksize];
    GraphElem step = gridDim.x;
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
        GraphWeight w = 0.;
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x+v_base];
        __syncthreads();

        GraphElem start = range[0]-e_base;
        GraphElem end = range[1]-e_base;

        //GraphElem start = edge_indices[i];
        //GraphElem end = edge_indices[i+1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
            w += weights[e];
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
            vertex_degree[i+v_base] = w;
        __syncthreads();
    }
}
#endif
//#if 0
template<const int WarpSize, const int BlockSize>
__global__
void sum_vertex_weights_kernel
(
    GraphWeight* __restrict__ vertex_weights, 
    GraphWeight* __restrict__ weights,
    GraphElem*   __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{   
    __shared__ GraphElem ranges[BlockSize/WarpSize*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem* t_ranges = &ranges[block_tid/WarpSize*2];

    GraphElem step = gridDim.x*BlockSize/WarpSize;
    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += step)
    {   
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();
       
        GraphElem start = t_ranges[0]-e_base;
        GraphElem   end = t_ranges[1]-e_base;
        GraphWeight w = 0.; 
        for(GraphElem e = start+warp_tid; e < end; e += WarpSize)
            w += weights[e];
        warp.sync();

        for(int i = warp.size()/2; i > 0; i/=2)
            w += warp.shfl_down(w, i);
 
        if(warp_tid == 0) 
            vertex_weights[u+v_base] = w;
        warp.sync();
    }
}
//#endif

#if 0
template<const int WarpSize, const int BlockSize>
__global__
void sum_vertex_weights_kernel
(
    GraphWeight* __restrict__ vertex_weights, 
    GraphWeight* __restrict__ weights,
    GraphElem*   __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{   
    __shared__ GraphElem ranges[BlockSize];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*WarpSize];

    GraphElem nu = (nv + gridDim.x*(BlockSize/WarpSize)-1) / (gridDim.x*(BlockSize/WarpSize));
    GraphElem u0 = (BlockSize/WarpSize*blockIdx.x+block_tid/WarpSize)*nu;
    nu = ((u0+nu > nv) ? nv-u0 : nu);

    GraphElem start, end;
    start = indices[u0+v_base]-e_base;

    for(GraphElem u = 0; u < nu; ++u)
    {
        if(u%WarpSize == 0)
        {
            warp.sync();
            if(u+warp_tid < nu)
                t_ranges[warp_tid] = indices[u0+u+warp_tid+v_base+1];
            warp.sync();
        }
        end = t_ranges[u%WarpSize]-e_base;

        GraphWeight w = 0.; 
        for(GraphElem e = start+warp_tid; e < end; e += WarpSize)
            w += weights[e];
        warp.sync();

        for(int i = warp.size()/2; i > 0; i/=2)
            w += warp.shfl_down(w, i);
 
        if(warp_tid == 0) 
            vertex_weights[u+u0+v_base] = w;
        start = end;
    }
}
#endif

void sum_vertex_weights_cuda
(
    GraphWeight* vertex_weights,
    GraphWeight* weights,
    GraphElem*   indices,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
     GraphElem nv = v1-v0;
     GraphElem nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE); 
     sum_vertex_weights_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(vertex_weights, weights, indices, v0, e0, nv);
}

template<const int BlockSize>
__global__
void compute_community_weighted_orders_kernel
(
    GraphWeight* __restrict__ commWeights,
    GraphElem*   __restrict__ commIds,
    GraphWeight* __restrict__ vertexWeights,
    const GraphElem nv
)
{
    GraphElem u0 = threadIdx.x+BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < nv; i += BlockSize*gridDim.x)
    {
        GraphElem comm_id = commIds[i];
        GraphWeight w = vertexWeights[i];
        atomicAdd(commWeights+comm_id, w);
    }
}

void compute_community_weights_cuda
(
    GraphWeight* commWeights,
    GraphElem* commIds, 
    GraphWeight* vertexWeights,
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    GraphElem nblocks = (nv > MAX_GRIDDIM) ? MAX_GRIDDIM : nv;
    compute_community_weighted_orders_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(commWeights, commIds, vertexWeights, nv);
}

template<const int BlockSize>
__global__ 
void singleton_partition_kernel
(
    GraphElem*   __restrict__ commIds,
    GraphWeight* __restrict__ commWeights,
    GraphWeight* __restrict__ vertexWeights,
    const GraphElem nv
)
{
    GraphElem u0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < nv; i += BlockSize*gridDim.x)
    {
        commIds[i] = i;
        commWeights[i] = vertexWeights[i];
    }    
}

void singleton_partition_cuda
(
    GraphElem* commIds, 
    GraphWeight* commWeights, 
    GraphWeight* vertexWeights, 
    const GraphElem& nv, 
    cudaStream_t stream = 0
)
{
    GraphElem nblocks = (nv > MAX_GRIDDIM) ? MAX_GRIDDIM : nv;
    singleton_partition_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(commIds, commWeights, vertexWeights, nv);
}

template<const int BlockSize>
__global__
void scan_edges_kernel
(
    GraphElem* __restrict__ edges,
    Edge*      __restrict__ edgeList,
    const GraphElem ne
)
{
    GraphElem u0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < ne; i += BlockSize*gridDim.x)
    {
        Edge e = edgeList[i];
        edges[i] = e.tail_;
    }
}

void scan_edges_cuda
(
    GraphElem* edges, 
    Edge* edgeList, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem ne = e1-e0;
    long long nblocks = (ne > MAX_GRIDDIM) ? MAX_GRIDDIM : ne;
    scan_edges_kernel<BLOCKDIM01><<<nblocks,BLOCKDIM01,0,stream>>>(edges, edgeList, ne);
}

template<const int BlockSize>
__global__
void scan_edge_weights_kernel
(
    GraphWeight* __restrict__ edgeWeights,
    Edge*        __restrict__ edgeList,
    const GraphElem ne
)
{
    GraphElem u0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < ne; i += BlockSize*gridDim.x)
    {
        Edge e = edgeList[i];
        edgeWeights[i] = e.weight_;
    }
}

void scan_edge_weights_cuda
(
    GraphWeight* edgeWeights, 
    Edge* edgeList, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem ne = e1-e0;
    long long nblocks = (ne > MAX_GRIDDIM) ? MAX_GRIDDIM : ne;
    scan_edge_weights_kernel<BLOCKDIM01><<<nblocks,BLOCKDIM01,0,stream>>>(edgeWeights, edgeList, ne);
}

template<const int WarpSize, const int BlockSize>
__global__
void max_vertex_weights_kernel
(
    GraphWeight* __restrict__ maxVertexWeights,
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   __restrict__ edge_indices,
    const GraphElem v_base,   
    const GraphElem e_base,
    const GraphElem nv 
)
{
    __shared__ GraphElem ranges[BlockSize/WarpSize*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem* t_ranges = &ranges[block_tid/WarpSize*2];

    GraphElem step = gridDim.x*BlockSize/WarpSize;
    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += step)
    {   
        if(warp_tid < 2)
            t_ranges[warp_tid] = edge_indices[u+warp_tid+v_base];
        warp.sync();
       
        GraphElem start = t_ranges[0]-e_base;
        GraphElem   end = t_ranges[1]-e_base;
        volatile GraphWeight w = 0.; 
        for(GraphElem e = start+warp_tid; e < end; e += WarpSize)
        {
            GraphWeight tmp = edgeWeights[e];
            w = (tmp > w) ? tmp : w;
        }
        warp.sync();

        for(int i = warp.size()/2; i > 0; i/=2)
        {
            GraphWeight tmp = warp.shfl_down(w, i);
            w = (tmp > w) ? tmp : w;
        }
        if(warp_tid == 0) 
            maxVertexWeights[u+v_base] = w;
        warp.sync();
    }
}

void max_vertex_weights_cuda
(
    GraphWeight* maxVertexWeights,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv > MAX_GRIDDIM) ? MAX_GRIDDIM : nv;
    max_vertex_weights_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(maxVertexWeights, edgeWeights, indices, v0, e0, nv);
}

template<const int BlockSize>
__global__
void max_order_reduce_kernel
(
    GraphElem* __restrict__ orders,
    GraphElem* __restrict__ indices, 
    GraphElem nv
)
{
    __shared__ GraphElem max_shared[BlockSize];

    max_shared[threadIdx.x] = 0;

    GraphElem u0 = threadIdx.x + BlockSize * blockIdx.x; 

    for(GraphElem u = u0; u < nv; u += BlockSize*gridDim.x)
    {    
        GraphElem order = indices[u+1]-indices[u];
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

template<const int BlockSize>
__global__
void max_order_kernel
(
    GraphElem* __restrict__ orders, 
    GraphElem nv
)
{
    __shared__ GraphElem max_shared[BlockSize];

    max_shared[threadIdx.x] = 0;

    GraphElem u0 = threadIdx.x;

    for(GraphElem u = u0; u < nv; u += BlockSize)
    {
        GraphElem order = orders[u];
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

GraphElem max_order_cuda
(
    GraphElem* indices,
    GraphElem nv, 
    cudaStream_t stream = 0  
)
{
    GraphElem* max_reduced;
    long long nblocks = (nv > MAX_GRIDDIM) ? MAX_GRIDDIM : nv;
    CudaMalloc(max_reduced, sizeof(GraphElem)*nblocks);
    max_order_reduce_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(max_reduced, indices, nv);
    max_order_kernel<BLOCKDIM01><<<1,BLOCKDIM01, 0, stream>>>(max_reduced, nblocks);
    GraphElem max;
    CudaMemcpyAsyncDtoH(&max, max_reduced, sizeof(GraphElem), 0);
    CudaFree(max_reduced);
    return max;
}

template<const int BlockSize>
__global__
void move_index_orders_kernel
(
    GraphElem* __restrict__ dest, 
    GraphElem* __restrict__ src, 
    const GraphElem n
)
{
    const int i0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = i0; i < n; i += BlockSize*gridDim.x)
        dest[i] = src[i]; 
}

void move_index_orders_cuda
(
    GraphElem* dest, 
    GraphElem* src, 
    const GraphElem& v0, 
    const GraphElem& v1, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem ne = e1-e0;
    long long nblocks = (ne > MAX_GRIDDIM) ? MAX_GRIDDIM : ne;
    move_index_orders_kernel<BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>(dest, src, ne);
}

template<const int WarpSize, const int BlockSize>
__global__
void reorder_edges_by_keys_kernel
(
    GraphElem* __restrict__ edges,
    GraphElem* indexOrders,
    GraphElem* __restrict__ indices,
    GraphElem* buff,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid  = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*2];

    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end = t_ranges[1]-e_base;
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edges[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
            edges[i] = buff[i];
        warp.sync();
    } 
}

void reorder_edges_by_keys_cuda
(
    GraphElem* edges, 
    GraphElem* indexOrders, 
    GraphElem* indices, 
    GraphElem* buff, 
    const GraphElem& v0, 
    const GraphElem& v1,  
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE);
    reorder_edges_by_keys_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(edges, indexOrders, indices, buff, v0, e0, nv);
}

template<const int WarpSize, const int BlockSize>
__global__
void reorder_weights_by_keys_kernel
(
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   indexOrders,
    GraphElem*   __restrict__ indices,
    GraphWeight* buff,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid  = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*2];

    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edgeWeights[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
            edgeWeights[i] = buff[i];
        warp.sync();
    } 
}

void reorder_weights_by_keys_cuda
( 
    GraphWeight* edgeWeights, 
    GraphElem* indexOrders, 
    GraphElem* indices , 
    GraphWeight* buff, 
    const GraphElem& v0, 
    const GraphElem& v1,  
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    GraphElem nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE);
    reorder_weights_by_keys_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01,0,stream>>>(edgeWeights, indexOrders, indices, buff, v0, e0, nv);
}
#if 0
template<const int WarpSize, const int BlockSize>
__global__
void build_local_commid_offset_kernel
(
    GraphElem* offset,
    GraphElem* edges,
    GraphElem* indices,
    GraphElem* commIds,
    GraphElem nv
)
{
    
}

void build_local_commid_offset_cuda
(
)
{

}

void compute_vertex_self_weight
(
    GraphWeight* vertexSelfWeights,
    GraphElem*   edges,
    GraphWeight* edgesWeights,
    GraphElem* indices,
    const GraphElem& nv,
    cosnt GraphElem& ne,
    cudaStream_t stream =0
)
{

}
#endif
#if 0
void compute_modularity_cuda
(
    GraphWeight* vertexSelfWeights,
    GraphWeight* commWeights,
    GraphWeight* m,
    const GraphElem& nv
)
{

}
#endif
#if 0
//#else

template<const int WarpSize, const int BlockSize>
__global__
void fill_edges_community_ids_kernel
(
    GraphElem2* __restrict__ commIdKeys,
    GraphElem*  __restrict__ edges,
    GraphElem*  __restrict__ indices,
    GraphElem*  __restrict__ commIds,
    const GraphElem v_base, 
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__  GraphElem ranges[BlockSize];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem* t_ranges = &ranges[(block_tid/WarpSize)*WarpSize];

    GraphElem nu = (nv + gridDim.x-1) / gridDim.x;
    GraphElem u0 = blockIdx.x*nu;
    GraphElem u1 = u0 + nu;
    if(u1 > nv) 
        u1 = nv;
    GraphElem start, end;
    start = indices[u0+v_base]-e_base;
    for(GraphElem u = 0; u < nu; ++u)
    {
        if(u%WarpSize == 0)
        {
            warp.sync();
            if(u+warp_tid <= nu)
                t_ranges[warp_tid] = indices[u0+u+warp_tid+v_base+1];
            warp.sync();
        }
        end = t_ragnes[u%WarpSize]-e_base;
        for(GraphElem i = start+warp_id; i < end; i += WarpSize)
        {
            GraphElem commId = commIds[edges[i]];    
            #ifdef USE_32BIT_GRAPH
            commIdKeys[i] = make_int2(u, commId);
            #else
            commIdKeys[i] = make_longlong2(u, commId);
            #endif
        }
        start = end;
    } 
}

void fill_edges_community_ids_cuda
(
    GraphElem2* commIdKeys, 
    GraphElem* edges,
    GraphElem* indices,
    GraphElem* commIds,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv/(BLOCKDIM01/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM01/WARPSIZE);    
    fill_edges_community_ids_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
    (commIdKeys, edges, indices, commIds, v0, e0, nv);
}
#endif
