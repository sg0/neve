#pragma once
#ifndef GRAPH_CUH_
#define GRAPH_CUH_

#ifdef USE_32_BIT_GRAPH
__device__ GraphWeight my_atomic_max(GraphWeight* address, GraphWeight val)
{
    unsigned *addr_as_ull = (unsigned*)address;
    unsigned old = *addr_as_ull;
    unsigned assumed;
    do 
    {
        assumed = old;
        //GraphWeight* ptr = (GraphWeight*)&assumed;
        if (val > __uint_as_float(assumed))
            old = atomicCAS(addr_as_ull, assumed,  __float_as_uint(val));
        else
            break;
    } while(assumed != old);
    return (GraphWeight)__uint_as_float((old);
}

#else 
__device__ GraphWeight my_atomic_max(GraphWeight* address, GraphWeight val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do 
    {
        assumed = old;
        if (val > __longlong_as_double(assumed))
            old = atomicCAS(addr_as_ull, assumed,(unsigned long long)__double_as_longlong(val));
        else
            break;
    } while(assumed != old);

    return (GraphWeight)__longlong_as_double(old);
}
#endif

template<const int blocksize>
__global__
void nbrscan_kernel(GraphWeight* __restrict__ edge_weights, Edge* 
__restrict__ edge_list, GraphElem* __restrict__ edge_indices,GraphElem nv)
{
    __shared__ GraphElem range[2];
    GraphElem step = gridDim.x;
    for(GraphElem i = 0; i < nv; i += step)
    {
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
	__syncthreads();

        GraphElem start = range[0];
        GraphElem end = range[1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
            edge_weights[e] = edge_list[e].weight_;
    } 
} 

template<const int blocksize>
__global__
void nbrsum_kernel(GraphWeight* __restrict__ vertex_degree, Edge* __restrict__ edge_list, 
GraphElem* __restrict__ edge_indices, GraphElem nv)
{
    __shared__ GraphElem range[2];
    GraphElem step = gridDim.x;
    for(GraphElem i = 0; i < nv; i += step)
    {
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
        __syncthreads();

        GraphElem start = range[0];
        GraphElem end = range[1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
            vertex_degree[i] += edge_list[e].weight_;
    } 
}

template<const int blocksize>
__global__
void nbrmax_kernel(GraphWeight* __restrict__ vertex_degree, Edge* __restrict__ edge_list, 
GraphElem* __restrict__ edge_indices, GraphElem nv)
{
    __shared__ GraphElem range[2];
    GraphElem step = gridDim.x;
    for(GraphElem i = 0; i < nv; i += step)
    {
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
	__syncthreads();

        GraphElem start = range[0];
        GraphElem end = range[1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
        {
            GraphWeight w = edge_list[e].weight_;
            my_atomic_max(vertex_degree+i, w);
        }
    } 
}

#endif
