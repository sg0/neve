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
        if (val > __uint_as_float(assumed))
            old = atomicCAS(addr_as_ull, assumed,  __float_as_uint(val));
        else
            break;
    } while(assumed != old);
    return (GraphWeight)__uint_as_float((old);
}

__device__ GraphWeight my_atomic_add(GraphWeight* address, GraphWeight val)
{
    unsigned *addr_as_ull = (unsigned*)address;
    unsigned old = *addr_as_ull;
    unsigned assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __float_as_unit(val+__uint_as_float(assumed)));
    } while(assumed != old);
    return (GraphWeight)__uint_as_float(old);
}

#else
__device__ GraphWeight my_atomic_add(GraphWeight* address, GraphWeight val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,__double_as_longlong(__longlong_as_double(assumed)+val));
    } while(assumed != old);

    return (GraphWeight)__longlong_as_double(old);
}

__device__ GraphWeight my_atomic_max(GraphWeight* address, GraphWeight val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do 
    {
        assumed = old;
        if (val > __longlong_as_double(assumed))
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
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
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
        if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
	__syncthreads();

        GraphElem start = range[0];
        GraphElem end = range[1];
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
    //__shared__ GraphElem range[2];
    volatile  __shared__ GraphWeight data[blocksize];
    GraphElem step = gridDim.x;
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
        /*if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
        __syncthreads();
        */
        //GraphElem start = range[0];
        //GraphElem end = range[1];
        data[threadIdx.x] = 0.;
        __syncthreads();
        GraphElem start = edge_indices[i];
        GraphElem end = edge_indices[i+1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
        {
            GraphWeight w = edge_list[e].weight_;
            data[threadIdx.x] += w;
        }
         __syncthreads();
        for (unsigned int s=blocksize/2; s>32; s>>=1) 
        {
            if (threadIdx.x < s)
                data[threadIdx.x] += data[threadIdx.x + s];
            __syncthreads();
        }
        if(threadIdx.x < 32)
            data[threadIdx.x] += data[threadIdx.x + 32];
            __syncthreads();
        if(threadIdx.x < 16)
            data[threadIdx.x] += data[threadIdx.x + 16];
            __syncthreads();
        if(threadIdx.x < 8)
            data[threadIdx.x] += data[threadIdx.x + 8];
            __syncthreads();
        if(threadIdx.x < 4)
            data[threadIdx.x] += data[threadIdx.x + 4];
            __syncthreads();
        if(threadIdx.x < 2)
            data[threadIdx.x] += data[threadIdx.x + 2];
            __syncthreads();
        if(threadIdx.x < 1)
            data[threadIdx.x] += data[threadIdx.x + 1]; 
            //__syncthreads();
        if(threadIdx.x == 0)
            vertex_degree[i] += data[0];
        __syncthreads();
    }
}

template<const int blocksize>
__global__
void nbrmax_kernel(GraphWeight* __restrict__ vertex_degree, Edge* __restrict__ edge_list, 
GraphElem* __restrict__ edge_indices, GraphElem nv)
{
    //__shared__ GraphElem range[2];
    volatile __shared__ GraphWeight data [blocksize];
    GraphElem step = gridDim.x;
    for(GraphElem i = blockIdx.x; i < nv; i += step)
    {
        /*if(threadIdx.x < 2)
            range[threadIdx.x] = edge_indices[i+threadIdx.x];
	__syncthreads();
        */
        //GraphElem start = range[0];
        //GraphElem end = range[1];
        data[threadIdx.x] = 0.;
        __syncthreads();

        GraphElem start = edge_indices[i];
        GraphElem end = edge_indices[i+1];
        for(GraphElem e = start+threadIdx.x; e < end; e += blocksize)
        {
            GraphWeight w = edge_list[e].weight_;
            //my_atomic_max(vertex_degree+i, w);
            if(data[threadIdx.x] < w)
                data[threadIdx.x] = w;
        }
        for (unsigned int s=blocksize/2; s>32; s>>=1)
        {
            if (threadIdx.x < s)
            {
                if(data[threadIdx.x + s] > data[threadIdx.x])
                    data[threadIdx.x] = data[threadIdx.x+s];
            }
            __syncthreads();
        }
        if(threadIdx.x < 32 && data[threadIdx.x+32] > data[threadIdx.x])
            data[threadIdx.x] = data[threadIdx.x+32];
        __syncthreads();
        if(threadIdx.x < 16 && data[threadIdx.x+16] > data[threadIdx.x])
            data[threadIdx.x] = data[threadIdx.x+16];
        __syncthreads();
        if(threadIdx.x < 8 && data[threadIdx.x+8] > data[threadIdx.x])
            data[threadIdx.x] = data[threadIdx.x+8];
        __syncthreads();
        if(threadIdx.x < 4 && data[threadIdx.x+4] > data[threadIdx.x])
            data[threadIdx.x] = data[threadIdx.x+4];
        __syncthreads();
        if(threadIdx.x < 2 && data[threadIdx.x+2] > data[threadIdx.x])
            data[threadIdx.x] = data[threadIdx.x+2];
        __syncthreads();
        if(threadIdx.x < 1 && data[threadIdx.x+1] > data[threadIdx.x])    
           data[threadIdx.x] = data[threadIdx.x+1];
        //__syncthreads();
        if(threadIdx.x == 0)
            vertex_degree[i] = data[0]; 
        __syncthreads();
    } 
}

#endif
