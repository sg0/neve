#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#include "graph_gpu.hpp"
#include "graph_cuda.hpp"
//include host side definition of graph
#include "graph.hpp"
#include "cuda_wrapper.hpp"

#ifdef CHECK
#include "graph_cpu.hpp"
#endif
GraphGPU::GraphGPU(Graph* graph) : graph_(graph), 
maxOrder_(0), nv_per_batch_(0), ne_per_batch_(0)//, mass_(0)
{
    nv_ = graph->get_num_vertices();
    ne_ = graph->get_num_edges();

    //alloc buffer
    CudaMalloc(indices_, sizeof(GraphElem)*(nv_+1));
    CudaMalloc(vertexWeights_, sizeof(GraphWeight)*nv_);
    CudaMalloc(commIds_, sizeof(GraphElem)*nv_);
    CudaMalloc(commWeights_, sizeof(GraphWeight)*nv_);
    CudaMalloc(newCommIds_, sizeof(GraphElem)*nv_);

    //register pinned memory
    indicesHost_     = graph_->get_index_ranges();
    edgeWeightsHost_ = graph_->get_edge_weights(); 
    edgesHost_       = graph_->get_edges(); 

    //CudaHostRegister(indicesHost_,     sizeof(GraphElem)*(nv_+1), cudaHostRegisterPortable);
    //CudaHostRegister(edgeWeightsHost_, sizeof(GraphWeight)*ne_,   cudaHostRegisterPortable);
    //CudaHostRegister(edgesHost_,       sizeof(GraphElem)*ne_,     cudaHostRegisterPortable);

    CudaMemcpyAsyncHtoD(indices_, indicesHost_, sizeof(GraphElem)*(nv_+1), 0);
    CudaMemset(vertexWeights_, 0, sizeof(GraphWeight)*nv_);
    CudaMemset(commWeights_, 0, sizeof(GraphWeight)*nv_);
    maxOrder_ = max_order();

    //std::cout << "max order is " << maxOrder_ << std::endl;

    unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight); 
    nv_per_batch_ = determine_optimal_vertices_per_batch(nv_, ne_, maxOrder_, unit_size); 
    ne_per_batch_ = determine_optimal_edges_per_batch (nv_, ne_, unit_size);

    //std::cout << ne_per_batch_ << std::endl;

    determine_optimal_vertex_partition(indicesHost_, nv_, ne_, ne_per_batch_, vertex_partition_);
    //std::cout << vertex_partition_.size() << std::endl;
    //for(int i = 0; i < vertex_partition_.size(); ++i)
    //    std::cout << vertex_partition_[i] << std::endl;

    CudaMalloc(edges_,       unit_size*ne_per_batch_);
    CudaMalloc(edgeWeights_, unit_size*ne_per_batch_);
    CudaMalloc(commIdKeys_,  sizeof(GraphElem2)*ne_per_batch_);
    CudaMalloc(indexOrders_, sizeof(GraphElem)*ne_per_batch_);

    /*#ifdef USE_PINNED_HOST
    CudaMallocHost(indexOrdersHost_, sizeof(GraphElem)*ne_per_batch_);
    #else      
    indexOrdersHost_ = new GraphElem [ne_per_batch_];
    #endif*/
    CudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    //CudaHostRegister(indexOrdersHost_, sizeof(GraphElem)*nv_per_batch_*maxOrder_, cudaHostRegisterPortable);

    sum_vertex_weights();
    compute_mass();
}

GraphGPU::~GraphGPU()
{
    CudaFree(edges_);
    CudaFree(edgeWeights_);
    CudaFree(commIdKeys_);
    CudaFree(indices_);
    CudaFree(vertexWeights_);
    CudaFree(commIds_);
    CudaFree(commWeights_);
    CudaFree(newCommIds_);
    CudaFree(indexOrders_);

    /*#ifdef USE_PINNED_HOST
    CudaFreeHost(indexOrdersHost_);
    #else
    delete [] indexOrdersHost_;
    #endif*/
    //CudaHostUnregister(indicesHost_);
    //CudaHostUnregister(edgesHost_);
    //CudaHostUnregister(edgeWeightsHost_);
    //CudaHostUnregister(indexOrdersHost_);
}

//TODO: in the future, this could be moved to a new class
GraphElem GraphGPU::determine_optimal_vertices_per_batch
(
    const GraphElem& nv, 
    const GraphElem& ne, 
    const GraphElem& maxOrder, 
    const unsigned& unit_size
)
{
    float free_m;//,total_m,used_m;
    size_t free_t,total_t;

    CudaCall(cudaMemGetInfo(&free_t,&total_t));

    float occ_m = (uint64_t)2*(1.5*sizeof(GraphElem)+sizeof(GraphWeight))*nv/1048576.0;
    free_m =(uint64_t)free_t/1048576.0 - occ_m;

    GraphElem nv_per_batch = (GraphElem)(free_m / unit_size / 7.5 / maxOrder * 1048576.0); //7 is the minimum, i chose 8

    return ((nv_per_batch > nv) ? nv : nv_per_batch);
}

GraphElem GraphGPU::determine_optimal_edges_per_batch
(
    const GraphElem& nv,
    const GraphElem& ne,
    const unsigned& unit_size
)
{
    float free_m;//,total_m,used_m;
    size_t free_t,total_t;

    CudaCall(cudaMemGetInfo(&free_t,&total_t));

    float occ_m = (uint64_t)2*(1.5*sizeof(GraphElem)+sizeof(GraphWeight))*nv/1048576.0;
    free_m =(uint64_t)free_t/1048576.0 - occ_m;

    GraphElem ne_per_batch = (GraphElem)(free_m / unit_size / 7.5 * 1048576.0); //7 is the minimum, i chose 8

    #ifdef CHECK
    //return 10103;
    return ((ne_per_batch > ne) ? ne/4 : ne_per_batch);
    #else
    return ((ne_per_batch > ne) ? ne : ne_per_batch);
    #endif
}

void GraphGPU::determine_optimal_vertex_partition
(
    GraphElem* indices,
    const GraphElem& nv,
    const GraphElem& ne,
    const GraphElem& ne_per_batch,
    std::vector<GraphElem>& vertex_partition
)
{
    vertex_partition.push_back(0);
    GraphElem start = 0;
    GraphElem end = 0;
    for(GraphElem idx = 1; idx <= nv; ++idx)
    {
        end = indices[idx];
        if(end - start > ne_per_batch)
        {
            vertex_partition.push_back(idx-1);
            start = indices[idx-1];
            idx--;
        }
    }
    vertex_partition.push_back(nv);
}

void GraphGPU::set_communtiy_ids(GraphElem* commIds)
{
    CudaMemcpyAsyncHtoD(commIds_, commIds, sizeof(GraphElem)*nv_, 0);
    compute_community_weights();
    CudaMemcpyAsyncHtoD(newCommIds_, commIds, sizeof(GraphElem)*nv_, 0);
}

void GraphGPU::compute_community_weights()
{
    compute_community_weights_cuda(commWeights_, commIds_, vertexWeights_, nv_);
}
#if 0
void GraphGPU::sort_edges_by_community_ids()
{
    cudaStream_t cuStreams[2];
    for(int i = 0; i < 2; ++i)
        cudaStreamCreate(&cuStreams[i]);

    //thrust::device_ptr<GraphElem>   edges_ptr = thrust::device_pointer_cast(edges_);
    thrust::device_ptr<GraphElem> orders_ptr = thrust::device_pointer_cast((GraphElem*)edgeWeights_);
    //thrust::device_ptr<GraphWeight> weights_ptr = thrust::device_pointer_cast(edgeWeights_);
    thrust::device_ptr<GraphElem2> keys_ptr = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;

    //GraphElem nbatches = (nv_+nv_per_batch_-1)/nv_per_batch_;
    //for(GraphElem i = 0; i < nbatches; ++i)
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    { 
        //GraphElem v0 = (i+0)*nv_per_batch_;
        //GraphElem v1 = (i+1)*nv_per_batch_;
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];

        //if(v1 > nv_) 
        //    v1 = nv_;
        //GraphElem nv = v1-v0;

        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;

        fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);

        CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*ne, cuStreams[0]);
        //CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[1]);

        fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);

        //fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);
        //fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);

        thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, comp);
         //thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, thrust::less<GraphElem2>());

        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphElem)*ne, cuStreams[1]);

        //fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);
 
        //thrust::sort_by_key(keys_ptr, keys_ptr+ne, edges_ptr, less_int2());
        move_index_orders_cuda((GraphElem*)commIdKeys_, (GraphElem*)edgeWeights_, v0, v1, e0, e1);

        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne,0);

        reorder_edges_by_keys_cuda((GraphElem*)edges_, (GraphElem*)commIdKeys_, indices_, ((GraphElem*)commIdKeys_)+ne, v0, v1,  e0, e1);
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, (GraphElem*)commIdKeys_, indices_, 
        (GraphWeight*)(((GraphElem*)commIdKeys_)+ne), v0, v1,  e0, e1);

        build_local_commid_offsets_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne,
        (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1);
        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 

        #ifdef CHECK
        //CudaDeviceSynchronize();
        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 
        CudaDeviceSynchronize();
        check_local_commid_offsets_cpu(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edgesHost_, 
        indicesHost_, commIds_, v0, v1, e0, e1, nv_);
        //print_results(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edgesHost_,indicesHost_, commIds_, v0, v1, e0, e1, nv_)
        #endif
        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 
        #if 0    
        CudaMemcpyAsyncDtoH(indexOrdersHost_, edgeWeights_, sizeof(GraphElem)*ne, cuStreams[0]);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, cuStreams[0]);
        
        //reorder the edges
        reorder_edges_by_keys_cuda((GraphElem*)edges_, (GraphElem*)edgeWeights_, indices_, v0, v1,  e0, e1); 

        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, cuStreams[0]);

        //reorder the weights
        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[0]);

        CudaMemcpyAsyncHtoD(edges_, indexOrdersHost_, sizeof(GraphElem)*ne, cuStreams[0]);
 
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, (GraphElem*)edges_, indices_, v0, v1, e0, e1);

        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        #endif
     }
     for(int i = 0; i < 2; ++i)
         cudaStreamDestroy(cuStreams[i]);
}
#endif
void GraphGPU::sort_edges_by_community_ids()
{
    cudaStream_t cuStreams[2];
    for(int i = 0; i < 2; ++i)
        cudaStreamCreate(&cuStreams[i]);

    thrust::device_ptr<GraphElem> orders_ptr = thrust::device_pointer_cast(indexOrders_);
    thrust::device_ptr<GraphElem2> keys_ptr = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;

    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    { 
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];

        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;

        fill_index_orders_cuda(indexOrders_, indices_, v0, v1, e0, e1, cuStreams[1]);

        CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*ne, cuStreams[0]);

        fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);

        thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, comp);

        //move_index_orders_cuda((GraphElem*)commIdKeys_, (GraphElem*)edgeWeights_, v0, v1, e0, e1);

        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne,0);

        reorder_edges_by_keys_cuda((GraphElem*)edges_, indexOrders_, indices_, (GraphElem*)commIdKeys_, v0, v1,  e0, e1);
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, indexOrders_, indices_, 
        (GraphWeight*)commIdKeys_, v0, v1,  e0, e1);

        build_local_commid_offsets_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne,
        (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1);

        #ifdef CHECK
        //CudaDeviceSynchronize();
        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 
        CudaDeviceSynchronize();
        check_local_commid_offsets_cpu(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edgesHost_, 
        indicesHost_, commIds_, v0, v1, e0, e1, nv_);
        //print_results(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edgesHost_,indicesHost_, commIds_, v0, v1, e0, e1, nv_)
        #endif
        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 
        #if 0    
        CudaMemcpyAsyncDtoH(indexOrdersHost_, edgeWeights_, sizeof(GraphElem)*ne, cuStreams[0]);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, cuStreams[0]);
        
        //reorder the edges
        reorder_edges_by_keys_cuda((GraphElem*)edges_, (GraphElem*)edgeWeights_, indices_, v0, v1,  e0, e1); 

        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, cuStreams[0]);

        //reorder the weights
        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[0]);

        CudaMemcpyAsyncHtoD(edges_, indexOrdersHost_, sizeof(GraphElem)*ne, cuStreams[0]);
 
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, (GraphElem*)edges_, indices_, v0, v1, e0, e1);

        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        #endif
     }
     for(int i = 0; i < 2; ++i)
         cudaStreamDestroy(cuStreams[i]);
}

void GraphGPU::singleton_partition()
{
    singleton_partition_cuda(commIds_, commWeights_, vertexWeights_, nv_);
    copy_vector_cuda(newCommIds_, commIds_, nv_);
}

GraphElem GraphGPU::max_order()
{
    return max_order_cuda(indices_, nv_);
}

void GraphGPU::sum_vertex_weights()
{
    //GraphElem nbatches = (nv_+nv_per_batch_-1)/nv_per_batch_;

    //cudaStream_t cuStreams[2];
    //cudaStreamCreate(&cuStreams[0]);
    //cudaStreamCreate(&cuStreams[1]);

    //for(GraphElem b = 0; b < nbatches; ++b)
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        //GraphElem v0 = b*nv_per_batch_;
        //GraphElem v1 = v0 + nv_per_batch_;
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];

        //v1 = (v1 > nv_) ? nv_: v1;

        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];

        //CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*(e1-e0), cuStreams[0]);
        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*(e1-e0), 0);
        //CudaDeviceSynchronize();
        sum_vertex_weights_cuda(vertexWeights_, (GraphWeight*)edgeWeights_, indices_, v0, v1, e0, e1);
        //CudaDeviceSynchronize();
    }

    //cudaStreamDestroy(cuStreams[0]);
    //cudaStreamDestroy(cuStreams[1]);  
}

void GraphGPU::scan_edges()
{
    void* edgeListHost = graph_->get_edge_list();
    //CudaHostRegister(edgeListHost, sizeof(Edge)*ne_,cudaHostRegisterPortable);
    //GraphElem nbatches = (nv_+nv_per_batch_-1)/nv_per_batch_;
    //for(GraphElem b = 0; b < nbatches; ++b)
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        //GraphElem v0 = b*nv_per_batch_;
        //GraphElem v1 = v0 + nv_per_batch_;
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];
        v1 = (v1 > nv_) ? nv_ : v1;
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        
        CudaMemcpyAsyncHtoD(commIdKeys_, ((Edge*)edgeListHost)+e0, sizeof(Edge)*(e1-e0),0);
        scan_edges_cuda((GraphElem*)edges_, (Edge*)commIdKeys_, e0, e1);
        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*(e1-e0), 0);
    }
    //CudaHostUnregister(edgeListHost);
}

void GraphGPU::scan_edge_weights()
{
    void* edgeListHost = graph_->get_edge_list();
    //CudaHostRegister(edgeListHost, sizeof(Edge)*ne_,cudaHostRegisterPortable);
    //GraphElem nbatches = (nv_+nv_per_batch_-1)/nv_per_batch_;
    //for(GraphElem b = 0; b < nbatches; ++b)
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        //GraphElem v0 = b*nv_per_batch_;
        //GraphElem v1 = v0 + nv_per_batch_;
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];
        //v1 = (v1 > nv_) ? nv_ : v1;
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        
        CudaMemcpyAsyncHtoD(commIdKeys_, ((Edge*)edgeListHost)+e0, sizeof(Edge)*(e1-e0),0);
        //CudaMemcpyHtoD(commIdKeys_, ((Edge*)edgeListHost)+e0, sizeof(Edge)*(e1-e0));
        scan_edge_weights_cuda((GraphWeight*)edgeWeights_, (Edge*)commIdKeys_, e0, e1);
        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*(e1-e0), 0);
        //CudaMemcpyDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*(e1-e0));
    }
    //CudaDeviceSynchronize();
    //CudaHostUnregister(edgeListHost);
}

void GraphGPU::max_vertex_weights()
{
    //GraphElem nbatches = (nv_+nv_per_batch_-1)/nv_per_batch_;
    //for(GraphElem b = 0; b < nbatches; ++b)
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        //GraphElem v0 = b*nv_per_batch_;
        //GraphElem v1 = v0 + nv_per_batch_;
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];
        //v1 = (v1 > nv_) ? nv_ : v1;
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        
        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*(e1-e0),0);
        max_vertex_weights_cuda(vertexWeights_, (GraphWeight*)edgeWeights_, indices_, v0, v1, e0, e1);
        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, vertexWeights_, sizeof(GraphWeight)*(v1-v0), 0);
    }
}
//#if 0
void GraphGPU::louvain_update()
{
    cudaStream_t cuStreams[2];
    for(int i = 0; i < 2; ++i)
        cudaStreamCreate(&cuStreams[i]);

    //thrust::device_ptr<GraphElem>   edges_ptr = thrust::device_pointer_cast(edges_);
    thrust::device_ptr<GraphElem> orders_ptr = thrust::device_pointer_cast((GraphElem*)edgeWeights_);
    //thrust::device_ptr<GraphWeight> weights_ptr = thrust::device_pointer_cast(edgeWeights_);
    thrust::device_ptr<GraphElem2> keys_ptr = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;

    //GraphElem nbatches = (nv_+nv_per_batch_-1)/nv_per_batch_;
    //for(GraphElem i = 0; i < nbatches; ++i)
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    { 
        //GraphElem v0 = (i+0)*nv_per_batch_;
        //GraphElem v1 = (i+1)*nv_per_batch_;
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];

        //if(v1 > nv_) 
        //    v1 = nv_;
        //GraphElem nv = v1-v0;

        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;

        fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);

        CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*ne, cuStreams[0]);
        //CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[1]);

        fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);

        //fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);
        //fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);

        thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, comp);
         //thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, thrust::less<GraphElem2>());

        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphElem)*ne, cuStreams[1]);

        //fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);
 
        //thrust::sort_by_key(keys_ptr, keys_ptr+ne, edges_ptr, less_int2());
        move_index_orders_cuda((GraphElem*)commIdKeys_, (GraphElem*)edgeWeights_, v0, v1, e0, e1);

        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne,0);

        reorder_edges_by_keys_cuda((GraphElem*)edges_, (GraphElem*)commIdKeys_, indices_, ((GraphElem*)commIdKeys_)+ne, v0, v1,  e0, e1);
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, (GraphElem*)commIdKeys_, indices_, 
        (GraphWeight*)(((GraphElem*)commIdKeys_)+ne), v0, v1,  e0, e1);

        build_local_commid_offsets_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne,
        (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1);
        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 

        #ifdef CHECK
        //CudaDeviceSynchronize();
        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 
        CudaDeviceSynchronize();
        check_local_commid_offsets_cpu(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edgesHost_, 
        indicesHost_, commIds_, v0, v1, e0, e1, nv_);
        //print_results(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edgesHost_,indicesHost_, commIds_, v0, v1, e0, e1, nv_)
        #endif

        louvain_update_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, (GraphElem*)edges_, (GraphWeight*)edgeWeights_,
                        indices_, vertexWeights_, commIds_, commWeights_, newCommIds_, mass_, v0, v1, e0, e1);
        #ifdef CHECK
        CudaDeviceSynchronize();
        check_louvain_update(edgesHost_, edgeWeightsHost_, indicesHost_,
        vertexWeights_, commIds_, commWeights_, newCommIds_,
        mass_, maxOrder_, v0, v1, nv_);
        #endif

        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, 0); 
        #if 0    
        CudaMemcpyAsyncDtoH(indexOrdersHost_, edgeWeights_, sizeof(GraphElem)*ne, cuStreams[0]);
        //CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, cuStreams[0]);
        
        //reorder the edges
        reorder_edges_by_keys_cuda((GraphElem*)edges_, (GraphElem*)edgeWeights_, indices_, v0, v1,  e0, e1); 

        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, cuStreams[0]);

        //reorder the weights
        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[0]);

        CudaMemcpyAsyncHtoD(edges_, indexOrdersHost_, sizeof(GraphElem)*ne, cuStreams[0]);
 
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, (GraphElem*)edges_, indices_, v0, v1, e0, e1);

        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, 0);
        #endif
     }
     for(int i = 0; i < 2; ++i)
         cudaStreamDestroy(cuStreams[i]);
}    


void GraphGPU::compute_mass()
{
    mass_ = compute_mass_cuda(vertexWeights_, nv_);
}

GraphWeight GraphGPU::compute_modularity()
{
    cudaStream_t cuStreams[2];
    for(int i = 0; i < 2; ++i)
        cudaStreamCreate(&cuStreams[i]);

    //thrust::device_ptr<GraphElem>   edges_ptr = thrust::device_pointer_cast(edges_);
    thrust::device_ptr<GraphElem> orders_ptr = thrust::device_pointer_cast((GraphElem*)edgeWeights_);
    //thrust::device_ptr<GraphWeight> weights_ptr = thrust::device_pointer_cast(edgeWeights_);
    thrust::device_ptr<GraphElem2> keys_ptr = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;
 
    GraphWeight* mod;
    CudaMalloc(mod, sizeof(GraphWeight)*MAX_GRIDDIM*BLOCKDIM01/8);
    CudaMemset(mod, 0, sizeof(GraphWeight)*MAX_GRIDDIM*BLOCKDIM01/8);

    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];
        //GraphElem nv = v1-v0;

        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;
        fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);

        CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*ne, cuStreams[0]);
        //CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[1]);

        fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);

        //fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);
        //fill_index_orders_cuda((GraphElem*)edgeWeights_, indices_, v0, v1, e0, e1, cuStreams[1]);

        thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, comp);
         //thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, thrust::less<GraphElem2>());

        //CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphElem)*ne, cuStreams[1]);

        //fill_edges_community_ids_cuda(commIdKeys_, (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1, cuStreams[0]);
 
        //thrust::sort_by_key(keys_ptr, keys_ptr+ne, edges_ptr, less_int2());
        move_index_orders_cuda((GraphElem*)commIdKeys_, (GraphElem*)edgeWeights_, v0, v1, e0, e1);

        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne,0);

        reorder_edges_by_keys_cuda((GraphElem*)edges_, (GraphElem*)commIdKeys_, indices_, ((GraphElem*)commIdKeys_)+ne, v0, v1,  e0, e1);
        reorder_weights_by_keys_cuda((GraphWeight*)edgeWeights_, (GraphElem*)commIdKeys_, indices_, 
        (GraphWeight*)(((GraphElem*)commIdKeys_)+ne), v0, v1,  e0, e1);

        build_local_commid_offsets_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne,
        (GraphElem*)edges_, indices_, commIds_, v0, v1, e0, e1);
 
        CudaLaunch((compute_modularity_reduce_cuda(mod, (GraphElem*)edges_, (GraphWeight*)edgeWeights_, indices_, 
        commIds_, commWeights_, ((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, mass_, v0, v1, e0, e1)));
    }

    GraphWeight q = compute_modularity_cuda(mod, MAX_GRIDDIM*BLOCKDIM01/8);

    CudaFree(mod);
    for(int i = 0; i < 2; ++i)
         cudaStreamDestroy(cuStreams[i]);

    #ifdef CHECK
    CudaDeviceSynchronize();
    GraphWeight q0 = check_modularity(edgesHost_, edgeWeightsHost_, indicesHost_, commIds_, commWeights_, mass_, nv_); 
    std::cout << "Modularities on GPU and CPU are " << q << " and " << q0 << std::endl;
    #endif
    return q;
}

void GraphGPU::move_edges_to_device(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream)
{
    GraphElem e0 = indicesHost_[v0];
    GraphElem e1 = indicesHost_[v1];
    GraphElem ne = e1-e0;

    CudaMemcpyAsyncHtoD(edges_, edgesHost_, sizeof(GraphElem)*ne, stream);
}

void GraphGPU::move_edges_to_host(const GraphElem& v0,  const GraphElem& v1, cudaStream_t stream)
{
    GraphElem e0 = indicesHost_[v0];
    GraphElem e1 = indicesHost_[v1];
    GraphElem ne = e1-e0;

    CudaMemcpyAsyncDtoH(edgesHost_, edges_, sizeof(GraphElem)*ne, stream);
}

void GraphGPU::move_weights_to_device(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream)
{
    GraphElem e0 = indicesHost_[v0];
    GraphElem e1 = indicesHost_[v1];
    GraphElem ne = e1-e0;

    CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_, sizeof(GraphWeight)*ne, stream);
}

void GraphGPU::move_weights_to_host(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream)
{
    GraphElem e0 = indicesHost_[v0];
    GraphElem e1 = indicesHost_[v1];
    GraphElem ne = e1-e0;

    CudaMemcpyAsyncDtoH(edgeWeightsHost_, edgeWeights_, sizeof(GraphWeight)*ne, stream);
}

void GraphGPU::memcpy()
{
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];

        move_edges_to_device(v0, v1);
        move_edges_to_host(v0, v1);
        move_weights_to_device(v0, v1);
        move_weights_to_host(v0, v1);
    }
}
//#endif
