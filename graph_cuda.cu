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
    GraphElem nblocks = (nv/(BLOCKDIM03/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM03/WARPSIZE); 
    CudaLaunch((reorder_weights_by_keys_kernel<WARPSIZE,BLOCKDIM03><<<nblocks, BLOCKDIM03,0,stream>>>(edgeWeights, indexOrders, indices, v0, e0, nv))); 
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
    long long nblocks = (nv/(BLOCKDIM03/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM03/WARPSIZE);
    CudaLaunch((reorder_edges_by_keys_kernel<WARPSIZE,BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>(edges, indexOrders, indices, v0, e0, nv)));
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
    long long nblocks = (nv/(BLOCKDIM03/WARPSIZE) > MAX_GRIDDIM) ? MAX_GRIDDIM : nv/(BLOCKDIM03/WARPSIZE);    
    CudaLaunch((fill_edges_community_ids_kernel<WARPSIZE, BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>
    (commIdKeys, edges, indices, commIds, v0, e0, nv)));
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
    CudaLaunch((fill_index_orders_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
    (indexOrders, indices, v0, e0, nv)));
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
        //warp.sync();

        for(int i = warp.size()/2; i > 0; i/=2)
            w += warp.shfl_down(w, i);
 
        if(warp_tid == 0) 
            vertex_weights[u+v_base] = w;
        //warp.sync();
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
    //cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    //const unsigned block_tid = block.thread_rank();
    //const unsigned warp_tid = warp.thread_rank();
    const unsigned lane_id = threadIdx.x & (WarpSize-1);

    GraphElem n = (nv+gridDim.x*BlockSize/WarpSize-1)/(gridDim.x*BlockSize/WarpSize);
    GraphElem u0 = (threadIdx.x/WarpSize+BlockSize/WarpSize*blockIdx.x)*n;
    GraphElem u1 = u0 + n;

    if(u1 > nv ) u1 = nv;
    u0 += v_base;
    u1 += v_base;

    GraphElem start = indices[u0]-e_base;
    GraphElem end = 0;
    for(GraphElem u = u0; u < u1; ++u)
    {   
        if(lane_id == 0)
            end = indices[u+1]-e_base;
        end = warp.shfl(end, 0);
       
        GraphWeight w = 0.; 
        for(GraphElem e = start+lane_id; e < end; e += WarpSize)
            w += weights[e];

        for(int i = warp.size()/2; i > 0; i/=2)
            w += warp.shfl_down(w, i);
 
        if(lane_id == 0) 
            vertex_weights[u] = w;
        //warp.sync();
        start = end;
    }
}
#endif
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
     CudaLaunch((sum_vertex_weights_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(vertex_weights, weights, indices, v0, e0, nv)));
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
    CudaLaunch((compute_community_weighted_orders_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(commWeights, commIds, vertexWeights, nv)));
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
    CudaLaunch((singleton_partition_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(commIds, commWeights, vertexWeights, nv)));
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
    CudaLaunch((scan_edges_kernel<BLOCKDIM01><<<nblocks,BLOCKDIM01,0,stream>>>(edges, edgeList, ne)));
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
    CudaLaunch((scan_edge_weights_kernel<BLOCKDIM01><<<nblocks,BLOCKDIM01,0,stream>>>(edgeWeights, edgeList, ne)));
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
    CudaLaunch((max_vertex_weights_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(maxVertexWeights, edgeWeights, indices, v0, e0, nv)));
}

//#if 0
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
//#endif
#if 0
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

    //GraphElem u0 = threadIdx.x + BlockSize * blockIdx.x; 
    GraphElem n = (nv + BlockSize*gridDim.x-1)/(BlockSize*gridDim.x);
    GraphElem u0 = (threadIdx.x + BlockSize*blockIdx.x)*n;
    //GraphElem u1 = u0 + n;
    n = (u0+n > nv) ? nv-u0 : n;
    //if(u1 > nv) u1 = nv;
    GraphElem start = indices[u0];
    GraphElem end;
    for(GraphElem u = 0; u < n; ++u)
    {    
        end = indices[u+u0+1];
        GraphElem order = end-start;
        if(max_shared[threadIdx.x] < order) 
            max_shared[threadIdx.x] = order;
        start = end;
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
#endif
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
    CudaLaunch((max_order_reduce_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(max_reduced, indices, nv)));
    CudaLaunch((max_order_kernel<BLOCKDIM01><<<1,BLOCKDIM01, 0, stream>>>(max_reduced, nblocks)));
    GraphElem max;
    CudaMemcpyAsyncDtoH(&max, max_reduced, sizeof(GraphElem), 0);
    CudaFree(max_reduced);
    return max;
}

template<typename T, const int BlockSize>
__global__
void copy_vector_kernel
(
    T* __restrict__ dest, 
    T* __restrict__ src, 
    const GraphElem n
)
{
    const int i0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = i0; i < n; i += BlockSize*gridDim.x)
        dest[i] = src[i]; 
}

void copy_vector_cuda
(
    GraphElem* dest,
    GraphElem* src,
    const GraphElem& ne_,
    cudaStream_t stream = 0
)
{
    GraphElem ne = ne_;
    long long nblocks = (ne > MAX_GRIDDIM) ? MAX_GRIDDIM : ne;
    CudaLaunch((copy_vector_kernel<GraphElem, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>(dest, src, ne)));
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
    CudaLaunch((copy_vector_kernel<GraphElem, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>(dest, src, ne)));
}
//#if 0
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

    //cg::thread_block block = cg::this_thread_block();
    //cg::thread_group warp = cg::tiled_partition<WarpSize>(block);
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    //const unsigned block_tid = block.thread_rank();
    //const unsigned warp_tid  = warp.thread_rank();
    const unsigned lane_id = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = threadIdx.x/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(lane_id < 2)
            t_ranges[lane_id] = indices[u+lane_id+v_base];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end = t_ranges[1]-e_base;
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edges[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edges[i] = buff[i];
        //warp.sync();
    } 
}
//#endif
#if 0
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
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    const unsigned lane_id = threadIdx.x & (WarpSize-1);

    GraphElem n = (nv+(BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem u0 = (threadIdx.x/WarpSize+BlockSize/WarpSize*blockIdx.x)*n;
    GraphElem u1 = u0 + n;
    if(u1 > nv) u1 = nv;
    u0 += v_base;
    u1 += v_base;
    GraphElem start = indices[u0]-e_base;
    GraphElem end;

    for(GraphElem u = u0; u < u1; ++u)
    {
        if(lane_id  == 0)
            end = indices[u+1]-e_base;
        end = warp.shfl(end, 0);

        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edges[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edges[i] = buff[i];
        start = end;
    } 
}
#endif

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
    CudaLaunch((reorder_edges_by_keys_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(edges, indexOrders, indices, buff, v0, e0, nv)));
}
//#if 0
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

    //cg::thread_block block = cg::this_thread_block();
    //cg::thread_group warp = cg::tiled_partition<WarpSize>(block);
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    //const unsigned block_tid = block.thread_rank();
    //const unsigned warp_tid  = warp.thread_rank();
    const unsigned lane_id = threadIdx.x & (WarpSize-1);
    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = threadIdx.x/WarpSize+BlockSize/WarpSize*blockIdx.x;
    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(lane_id < 2)
            t_ranges[lane_id] = indices[u+lane_id+v_base];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edgeWeights[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edgeWeights[i] = buff[i];
    } 
}
//#endif
#if 0
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
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned lane_id = threadIdx.x & (WarpSize-1);

    GraphElem n = (nv+(BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x); 
    GraphElem u0 = (threadIdx.x/WarpSize+BlockSize/WarpSize*blockIdx.x)*n;
    GraphElem u1 = u0 + n;
    if(u1 > nv) u1 = nv;
    u0 += v_base;
    u1 += v_base;
    GraphElem start = indices[u0]-e_base;
    GraphElem end;

    for(GraphElem u = u0; u < u1; ++u)
    {
        if(lane_id  == 0)
            end = indices[u+1]-e_base;
        end = warp.shfl(end, 0);

        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edgeWeights[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edgeWeights[i] = buff[i];
        start = end;
    } 
}
#endif

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
    CudaLaunch((reorder_weights_by_keys_kernel<WARPSIZE,BLOCKDIM01><<<nblocks, BLOCKDIM01,0,stream>>>(edgeWeights, indexOrders, indices, buff, v0, e0, nv)));
}


#if 0
template<const int BlockSize>
__global__
void build_local_commid_offsets_kernel
(
    GraphElem* localOffsets,
    GraphElem* localCommNums,
    GraphElem*   __restrict__ edges,
    GraphElem*   __restrict__ indices,
    GraphElem*   __restrict__ commIds,
    GraphElem v_base,
    GraphElem e_base,
    GraphElem nv
)
{
    GraphElem v0 = threadIdx.x+BlockSize*blockIdx.x+v_base;
    for(GraphElem v = v0; v < nv+v_base; v += BlockSize*gridDim.x)
    {
        GraphElem start = indices[v+0]-e_base;
        GraphElem end =   indices[v+1]-e_base;

        GraphElem count = 1;
        GraphElem target = commIds[edges[start]];
        localOffsets[start] = 0;

        for(GraphElem u = start+1; u < end; ++u)
        {
            GraphElem e = edges[u];
            GraphElem commId = commIds[e];
            if(commId != target)
            {
                localOffsets[count+start] = u-start;
                count++;
                target = commId;
            }
        }
        localCommNums[v-v_base] = count;
    }
}
#endif
//#if 0
template<const int BlockSize, const int WarpSize=32>
__global__
void build_local_commid_offsets_kernel
(
    GraphElem* localOffsets,
    GraphElem* localCommNums,
    GraphElem*   __restrict__ edges,
    GraphElem*   __restrict__ indices,
    GraphElem*   __restrict__ commIds,
    GraphElem v_base,
    GraphElem e_base,
    GraphElem nv
)
{  
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    unsigned lane_id = threadIdx.x &(WarpSize-1);
    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (threadIdx.x / WarpSize + (BlockSize/WarpSize)*blockIdx.x)*v1;
    v1 += v0;
    if(v1 > nv) v1 = nv;

    v0 += v_base;
    v1 += v_base;

    GraphElem start = indices[v0]-e_base;
    GraphElem end = 0;
    for(GraphElem v = v0; v < v1; ++v)
    {
        if(lane_id == 0)
            end   = indices[v+1]-e_base;
        end   = warp.shfl(end, 0);
        
        volatile GraphElem count = 0;
        volatile GraphElem target;
        volatile GraphElem localId = 0;

        while(count < end-start)
        {
            if(lane_id == 0x00)
                localOffsets[start+localId] = count; 
            target = commIds[edges[start+count]];
            volatile unsigned localCount = 0;
            for(GraphElem u = start+count; u < end; u += WarpSize)
            {
                if((u+lane_id) < end)
                {
                    if(commIds[edges[u+lane_id]] == target)
                        localCount++;
                    else
                        break;
                }
            }
            #pragma unroll
            for(int i = WarpSize/2; i > 0; i/=2)
                localCount += warp.shfl_down(localCount, i);
            count += localCount;
            count = warp.shfl(count, 0);
            localId++;
        }
        start = end;
        if(lane_id == 0x00)
            localCommNums[v-v_base] = localId;
    }
}
//#endif

void build_local_commid_offsets_cuda
(
    GraphElem* localOffsets,
    GraphElem* localCommNums,
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
    //long long nblocks = (nv+MAX_BLOCKDIM-1)/MAX_BLOCKDIM;
    //nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    long long nblocks = (nv+(BLOCKDIM03/4-1))/(BLOCKDIM03/4);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((build_local_commid_offsets_kernel<BLOCKDIM03,4><<<nblocks, BLOCKDIM03, 0, stream>>>
    (localOffsets,localCommNums, edges, indices, commIds, v0, e0, nv)));
}
//#if 0
template<const int BlockSize>
__global__
void update_commids_kernel
(
    GraphElem*   __restrict__ commIds,
    GraphElem*   __restrict__ newCommIds,
    GraphWeight* __restrict__ commWeights,
    GraphWeight* __restrict__ vertexWeights,
    const GraphElem v0,
    const GraphElem v1
)
{
    for(GraphElem v = threadIdx.x + BlockSize*blockIdx.x+v0; v < v1; v += BlockSize*gridDim.x)
    {
        GraphElem src = commIds[v];
        GraphElem dest = newCommIds[v];
        if(src != dest)
        {
            GraphWeight ki = vertexWeights[v];
            atomicAdd(commWeights+src, -ki);
            atomicAdd(commWeights+dest, ki);
            commIds[v] = dest;        
        } 
    }
}

void update_commids_cuda
(
    GraphElem* commIds,
    GraphElem* newCommIds,
    GraphWeight* commWeights,    
    GraphWeight* vertexWeights,
    const GraphElem& v0,
    const GraphElem& v1,
    cudaStream_t stream = 0
)
{
    long long nblocks = (v1-v0)/BLOCKDIM01;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((update_commids_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(commIds,newCommIds,commWeights, vertexWeights, v0, v1)));
}
//#if 0
template<const int BlockSize, const int WarpSize, const int TileSize>
__global__
void louvain_update_kernel
(
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    GraphElem*   edges,
    GraphWeight* edgeWeights,
    GraphElem*   indices,
    GraphWeight* vertexWeights,
    GraphElem*   commIds,
    GraphWeight* commWeights,
    GraphElem*   newCommIds,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphWeight self_shared[BlockSize/WarpSize];
    __shared__ Float2 gain_shared[BlockSize/TileSize];

    GraphWeight gain, selfWeight;
    Float2 target;

    //cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(warp);

    const unsigned warp_id      = threadIdx.x / WarpSize;
    const unsigned lane_id      = threadIdx.x & (WarpSize-1);
    const unsigned tile_id      = lane_id / TileSize;
    const unsigned tile_lane_id = threadIdx.x & (TileSize-1);

    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (warp_id + (BlockSize/WarpSize)*blockIdx.x)*v1;

    v1 += v0;
    if(v1 > nv) v1 = nv;
    v0 += v_base;
    v1 += v_base;
    GraphElem start, end;
    start = indices[v0]-e_base;
    
    for(GraphElem v = v0; v < v1; ++v)
    {
        selfWeight = 0;
        #ifdef USE_32BIT_GRAPH
        target = make_float2(-MAX_FLOAT, __int_as_float(0));
        #else
        target = make_double2(-MAX_FLOAT, __longlong_as_double(0LL));
        #endif

        if(lane_id == 0x00)
            self_shared[warp_id] = 0;

        if(lane_id == 0x00)
            end = indices[v+1]-e_base;
        end = warp.shfl(end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[v];
        GraphWeight ki = vertexWeights[v];
        //loop throught unique community ids
        for(GraphElem j = tile_id; j < localCommNum; j += WarpSize/TileSize)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            gain = 0.;
            GraphElem destCommId = commIds[edges[n0]];
            //tile.sync();
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                {
                    if(edges[k] != v)
                        selfWeight += edgeWeights[k];
                }
                //tile.sync();
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    selfWeight += tile.shfl_down(selfWeight, i);
                //tile.sync();
                if(tile_lane_id == 0x00)
                    self_shared[warp_id] = selfWeight;
                //tile.sync();
            }
            else
            {          
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                    gain += edgeWeights[k];
                //tile.sync();
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    gain += tile.shfl_down(gain, i);
                //tile.sync();
                if(tile_lane_id == 0x00)
                {
                    gain -= ki*commWeights[destCommId]/(2.*mass);
                    gain /= mass;
                    if(target.x < gain)
                    {
                        target.x = gain;
                        target.y = __longlong_as_double(destCommId);
                    }
                }
                //tile.sync();
            }
            //warp.sync();
            //tile.sync();
        }
        //warp.sync();

        if(tile_lane_id == 0x00)
            gain_shared[(WarpSize/TileSize)*warp_id+tile_id] = target;
        warp.sync();
        
        #pragma unroll
        for(unsigned int i = WarpSize/(TileSize*2); i > 0; i/=2)
        {
            if(lane_id < i)
            {
                if(gain_shared[(WarpSize/TileSize)*warp_id+lane_id+i].x > 
                   gain_shared[(WarpSize/TileSize)*warp_id+lane_id+0].x)
                {
                    gain_shared[(WarpSize/TileSize)*warp_id+lane_id+0] = 
                    gain_shared[(WarpSize/TileSize)*warp_id+lane_id+i];
                }
            }
            warp.sync();
        }
        //warp.sync();
        if(lane_id == 0)
        {
            /*Float2 choice = gain_shared[(WarpSize/TileSize)*warp_id];
            for(unsigned int i = 1; i < WarpSize/TileSize; ++i)
            {
                if(choice.x < gain_shared[(WarpSize/TileSize)*warp_id+i].x)
                    choice = gain_shared[(WarpSize/TileSize)*warp_id+i];
            }
            gain = choice.x;*/
            gain = gain_shared[(WarpSize/TileSize)*warp_id].x;
            localCommNum = __double_as_longlong(gain_shared[(WarpSize/TileSize)*warp_id].y);
            //localCommNum = __double_as_longlong(choice.y);
            selfWeight = self_shared[warp_id];
            selfWeight -= ki*(commWeights[myCommId]-ki)/(2.*mass); 
            selfWeight /= mass;
            gain -= selfWeight;
            if(gain > 0)
                newCommIds[v] = localCommNum;
        }
        warp.sync();
        start = end;
    }
}
//#endif
#if 0
template<const int WarpSize, const int TileSize>
__global__
void louvain_update_kernel
(
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    GraphElem*   edges,
    GraphWeight* edgeWeights,
    GraphElem*   indices,
    GraphWeight* vertexWeights,
    GraphElem*   commIds,
    GraphWeight* commWeights,
    GraphElem*   newCommIds,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    //__shared__ GraphWeight self_shared[1];
    __shared__ Float2 gain_shared[WarpSize/TileSize];

    GraphWeight gain, selfWeight;
    Float2 target;

    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(cg::this_thread_block());

    const unsigned tile_id      = threadIdx.x / TileSize;
    const unsigned tile_lane_id = threadIdx.x & (TileSize-1);
    GraphElem v0 = blockIdx.x+v_base;

    GraphElem start, end;
    start = indices[v0]-e_base;
    
    for(GraphElem v = v0; v < nv+v_base; v += gridDim.x)
    {
        selfWeight = 0;
        #ifdef USE_32BIT_GRAPH
        target = make_float2(-MAX_FLOAT, __int_as_float(0));
        #else
        target = make_double2(-MAX_FLOAT, __longlong_as_double(0LL));
        #endif

        if(threadIdx.x == 0x00)
            end = indices[v+1]-e_base;
        end = __shfl_sync(0xffffffff, end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[v];
        GraphWeight ki = vertexWeights[v];
        //loop throught unique community ids
        //__syncthreads();
        for(GraphElem j = tile_id; j < localCommNum; j += WarpSize/TileSize)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            gain = 0.;
            GraphElem destCommId = commIds[edges[n0]];
            tile.sync();
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                {
                    if(edges[k] != v)
                        selfWeight += edgeWeights[k];
                }
                //tile.sync();
                /*for(unsigned int i = TileSize/2; i > 0; i/=2)
                    selfWeight += tile.shfl_down(selfWeight, i);
                //tile.sync();
                if(tile_lane_id == 0x00)
                    self_shared[warp_id] = selfWeight;*/
                //tile.sync();
            }
            else
            {          
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                    gain += edgeWeights[k];
                //tile.sync();
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    gain += tile.shfl_down(gain, i);
                //tile.sync();
                if(tile_lane_id == 0x00)
                {
                    gain -= ki*commWeights[destCommId]/(2.*mass);
                    gain /= mass;
                    if(target.x < gain)
                    {
                        target.x = gain;
                        target.y = __longlong_as_double(destCommId);
                    }
                }
                //tile.sync();
            }
            //warp.sync();
            tile.sync();
        }
        //warp.sync();
         __syncthreads();
        for(unsigned i = WarpSize/2; i > 0; i/=2)
            selfWeight += __shfl_down_sync(0xffffffff, selfWeight, i);
        if(tile_lane_id == 0x00)
            gain_shared[tile_id] = target;
        //warp.sync();
        __syncthreads();
        #pragma unroll
        for(unsigned int i = WarpSize/(TileSize*2); i > 0; i/=2)
        {
            if(threadIdx.x < i)
            {
                if(gain_shared[threadIdx.x+i].x > 
                   gain_shared[threadIdx.x+0].x)
                {
                    gain_shared[threadIdx.x+0] = 
                    gain_shared[threadIdx.x+i];
                }
            }
            __syncthreads();
        }
        //warp.sync();
        if(threadIdx.x == 0)
        {
            /*Float2 choice = gain_shared[(WarpSize/TileSize)*warp_id];
            for(unsigned int i = 1; i < WarpSize/TileSize; ++i)
            {
                if(choice.x < gain_shared[(WarpSize/TileSize)*warp_id+i].x)
                    choice = gain_shared[(WarpSize/TileSize)*warp_id+i];
            }
            gain = choice.x;*/
            gain = gain_shared[0].x;
            localCommNum = __double_as_longlong(gain_shared[0].y);
            //localCommNum = __double_as_longlong(choice.y);
            //selfWeight = self_shared[warp_id];
            selfWeight -= ki*(commWeights[myCommId]-ki)/(2.*mass); 
            selfWeight /= mass;
            gain -= selfWeight;
            if(gain > 0)
                newCommIds[v] = localCommNum;
        }
        //warp.sync();
        start = end;
         __syncthreads();
    }
}
#endif

void louvain_update_cuda
(
    GraphElem* localCommOffsets, 
    GraphElem* localCommNums, 
    GraphElem*   edges,
    GraphWeight* edgeWeights,
    GraphElem*   indices, 
    GraphWeight* vertexWeights, 
    GraphElem*   commIds, 
    GraphWeight* commWeights, 
    GraphElem*   newCommIds,
    const GraphWeight& mass, 
    const GraphElem& v0, 
    const GraphElem& v1, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    const GraphElem nv = v1-v0;
    long long nblocks = nv / (BLOCKDIM01/WARPSIZE);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((louvain_update_kernel<BLOCKDIM02,WARPSIZE,4><<<nblocks, BLOCKDIM02, 0, stream>>>
    (localCommOffsets, localCommNums, edges, edgeWeights, indices, vertexWeights, commIds, commWeights, newCommIds, mass, v0, e0, v1-v0)));
    /*long long nblocks = nv;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((louvain_update_kernel<WARPSIZE,4><<<nblocks, WARPSIZE, 0, stream>>>
    (localCommOffsets, localCommNums, edges, edgeWeights, indices, vertexWeights, commIds, commWeights, newCommIds, mass, v0, e0, nv)));
    */
}

template<const int WarpSize=32>
__global__
void compute_mass_reduce_kernel
(
    GraphWeight* __restrict__ mass, 
    GraphWeight* __restrict__ vertexWeights, 
    GraphElem nv
)
{ 
    GraphWeight m = 0.;
    for(GraphElem i = threadIdx.x+WarpSize*blockIdx.x; i < nv; i += WarpSize*gridDim.x)
        m += vertexWeights[i];
    for(unsigned int i = WarpSize/2; i > 0; i/=2)
        m += __shfl_down_sync(0xffffffff, m, i, WarpSize);

    if(threadIdx.x ==0)
        mass[blockIdx.x] = m;
}

template<const int WarpSize=32>
__global__
void reduce_vector_kernel
(
    GraphWeight* mass,
    //GraphWeight* vertexWeights,
    GraphElem nv
)
{
    __shared__ GraphWeight m_shared[WarpSize];
    GraphWeight m = 0.;
    for(GraphElem i = threadIdx.x; i < nv; i += WarpSize*WarpSize)
        m += mass[i];
    for(unsigned int i = WarpSize/2; i > 0; i/=2)
        m += __shfl_down_sync(0xffffffff, m, i, WarpSize);

    if((threadIdx.x & (WarpSize-1)) == 0)
        m_shared[threadIdx.x/WarpSize] = m;
    __syncthreads();
    if(threadIdx.x / WarpSize == 0)
    {
        m = m_shared[threadIdx.x & (WarpSize-1)];
        for(unsigned int i = WarpSize/2; i > 0; i/=2)
            m += __shfl_down_sync(0xffffffff, m, i, WarpSize);
    }
    if(threadIdx.x ==0)
        *mass = m;
}

GraphWeight compute_mass_cuda
(
    GraphWeight* vertexWeights,
    GraphElem nv,
    cudaStream_t stream = 0
)
{
    GraphWeight *mass;
    const int nblocks = 4096;
    CudaMalloc(mass, sizeof(GraphWeight)*nblocks);
    CudaLaunch((compute_mass_reduce_kernel<WARPSIZE><<<nblocks, WARPSIZE>>>(mass, vertexWeights, nv)));
    CudaLaunch((reduce_vector_kernel<WARPSIZE><<<1, WARPSIZE*WARPSIZE>>>(mass, nblocks)));
    GraphWeight m;
    CudaMemcpyDtoH(&m, mass, sizeof(GraphWeight));
    CudaFree(mass);
    return 0.5*m;
}
//#if 0
template<const int BlockSize, const int WarpSize>
__global__
void compute_modularity_reduce_kernel
(
    GraphWeight* mod,
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    GraphElem* commIds,
    GraphWeight* commWeights,
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    //__shared__ GraphWeight self_shared[BlockSize/WarpSize];
    GraphWeight selfWeight;

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned warp_id      = threadIdx.x / WarpSize;
    const unsigned lane_id      = threadIdx.x & (WarpSize-1);

    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (warp_id + (BlockSize/WarpSize)*blockIdx.x)*v1;

    v1 += v0;
    if(v1 > nv) v1 = nv;
    v0 += v_base;
    v1 += v_base;
    GraphElem start, end;
    start = indices[v0]-e_base;
    
    for(GraphElem v = v0; v < v1; ++v)
    {
        selfWeight = 0;

        if(lane_id == 0x00)
            end = indices[v+1]-e_base;
        end = warp.shfl(end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[v];
        //loop throught unique community ids
        for(GraphElem j = 0; j < localCommNum; ++j)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            GraphElem destCommId = commIds[edges[n0]];
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+lane_id; k < n1; k+=WarpSize)
                    selfWeight += edgeWeights[k];
                for(unsigned int i = WarpSize/2; i > 0; i/=2)
                    selfWeight += warp.shfl_down(selfWeight, i);
                break;
            }
        }
        //warp.sync();
        if(lane_id == 0)
        {
            //selfWeight = self_shared[warp_id];
            selfWeight /= (2.*mass);
            GraphWeight ac = commWeights[v]; 
            selfWeight -= ac*ac/(4*mass*mass);
            mod[warp_id+BlockSize/WarpSize*blockIdx.x] += selfWeight;
        }
        warp.sync();
        start = end;
    }
}

void compute_modularity_reduce_cuda
(
    GraphWeight* mod,
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    GraphElem* commIds,
    GraphWeight* commWeights,
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1 - v0;
    long long nblocks = (nv+(BLOCKDIM02/WARPSIZE)-1)/(BLOCKDIM02/WARPSIZE);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((compute_modularity_reduce_kernel<BLOCKDIM02, WARPSIZE><<<nblocks, BLOCKDIM02, 0, stream>>>
    (mod, edges, edgeWeights, indices, commIds, commWeights, localCommOffsets, localCommNums, mass, v0, e0, nv)));
}

GraphWeight compute_modularity_cuda
(
    GraphWeight* mod,
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    CudaLaunch((reduce_vector_kernel<WARPSIZE><<<1, WARPSIZE*WARPSIZE, 0, stream>>>(mod, nv)));

    GraphWeight m;
    CudaMemcpyDtoH(&m, mod, sizeof(GraphWeight));
    return m;
}

//#endif
//#endif
#if 0
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
