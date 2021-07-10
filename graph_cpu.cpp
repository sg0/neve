#include <iostream>
#include <cstdlib>
#include <cmath>
#include <set>
#include <unordered_map>
#include "types.hpp"
#include "cuda_wrapper.hpp"

void check_local_commid_offsets_cpu
(
    GraphElem* localCommOffsets_dev, 
    GraphElem* localCommNums_dev, 
    GraphElem* edges, 
    GraphElem* indices,
    GraphElem* commIds_dev, 
    const GraphElem& v0, 
    const GraphElem& v1, 
    const GraphElem& e0, 
    const GraphElem& e1,
    const GraphElem& nv
)
{
    cudaStream_t cuStreams[3];
    
    for(int i = 0; i < 3; ++i)
        cudaStreamCreate(&cuStreams[i]);
 
    GraphElem *localCommOffsets_host, *localCommNums_host;
    GraphElem* commIds_host;

    GraphElem *localCommOffsets, *localCommNums;

    CudaMallocHost(localCommOffsets_host, sizeof(GraphElem)*(e1-e0));
    CudaMallocHost(localCommNums_host, sizeof(GraphElem)*(v1-v0));
    CudaMallocHost(commIds_host, sizeof(GraphElem)*nv);

    localCommOffsets = new GraphElem [e1-e0];
    localCommNums = new GraphElem [v1-v0];

    CudaMemcpyAsyncDtoH(localCommOffsets_host, localCommOffsets_dev, sizeof(GraphElem)*(e1-e0), cuStreams[0]);
    CudaMemcpyAsyncDtoH(localCommNums_host, localCommNums_dev, sizeof(GraphElem)*(v1-v0), cuStreams[1]);
    CudaMemcpyAsyncDtoH(commIds_host, commIds_dev, sizeof(GraphElem)*nv, cuStreams[2]);
    cudaStreamSynchronize(cuStreams[2]);

    for(GraphElem i = v0; i < v1; ++i)
    {
        GraphElem start, end;
        start = indices[i];
        end   = indices[i+1];
        //std::cout << start << " " << end << " " << end-start << std::endl;
        GraphElem target = commIds_host[edges[start]];
        localCommOffsets[start-e0] = 0;
        //GraphElem count = 1;
        GraphElem localId = 1;
        std::set<GraphElem> id_set;
        id_set.insert(target);
        for(GraphElem j = start+1; j < end; ++j)
        {
            GraphElem commId = commIds_host[edges[j]];
            id_set.insert(commId);
            //if(i-v0 == 0)
            //    std::cout << j-start << " " << commId << std::endl;
            if(commId != target)
            {
                target = commId;
                localCommOffsets[start-e0+localId] = j-start;
                localId++;
            }
        }
        //if(i-v0 == 0)
        //    std::cout << localId << std::endl;
        //localCommNums[i-v0] = localId;
        localCommNums[i-v0] = id_set.size();
    }
    CudaDeviceSynchronize();

    double err = 0;
    for(GraphElem i = 0; i < v1-v0; ++i)
        err += std::pow(localCommNums[i]-localCommNums_host[i], 2);
        //std::cout << localCommNums[i] << " " << localCommNums_host[i] << std::endl;
    if(err != 0.)
        std::cout << "There are errors in the numbers of the unique community IDs for each vertex.\n";// << err << std::endl;
    else
        std::cout << "The results for the numbers of the unique community IDs are correct.\n";

    for(GraphElem i = 0; i < v1-v0; ++i)
    {
        GraphElem start, num;
        start = indices[i+v0]-e0;
        num = localCommNums[i];

        GraphWeight err = 0.;
        for(GraphElem j = 0; j < num; ++j)
            err += std::pow(localCommOffsets[j+start]-localCommOffsets_host[j+start],2);
        if(err != 0.)
        {
            std::cout << err << std::endl;
            std::cout << "Index " << i+v0 << " has errors" << std::endl;
            exit(-1); 
        }
    }

    CudaFreeHost(localCommOffsets_host);  
    CudaFreeHost(localCommNums_host);
    CudaFreeHost(commIds_host);

    delete [] localCommOffsets;
    delete [] localCommNums;

    for(int i = 0; i < 3; ++i)
         cudaStreamDestroy(cuStreams[i]);
}

void check_louvain_update
(
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    GraphWeight* vertexWeights_dev,
    GraphElem*   commIds_dev,
    GraphWeight* commWeights_dev,
    GraphElem*   newCommIds_dev,
    const GraphWeight& mass,
    const GraphElem& max_order,
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& nv
)
{
    cudaStream_t cuStreams[4];

    cudaStreamCreate(&cuStreams[0]);
    cudaStreamCreate(&cuStreams[1]);
    cudaStreamCreate(&cuStreams[2]);
    cudaStreamCreate(&cuStreams[3]);

    GraphElem* commIds_host;
    GraphWeight* commWeights_host;
    GraphElem* newCommIds_host;
    GraphWeight* vertexWeights_host; 

    CudaMallocHost(commIds_host,       sizeof(GraphElem)*nv);
    CudaMallocHost(commWeights_host,   sizeof(GraphWeight)*nv);
    CudaMallocHost(newCommIds_host,    sizeof(GraphElem)*nv);
    CudaMallocHost(vertexWeights_host, sizeof(GraphWeight)*nv);

    CudaMemcpyAsyncDtoH(commIds_host, commIds_dev, sizeof(GraphElem)*nv, cuStreams[0]);
    CudaMemcpyAsyncDtoH(commWeights_host, commWeights_dev, sizeof(GraphWeight)*nv, cuStreams[1]);
    CudaMemcpyAsyncDtoH(vertexWeights_host, vertexWeights_dev, sizeof(GraphWeight)*nv, cuStreams[2]);
    CudaMemcpyAsyncDtoH(newCommIds_host, newCommIds_dev, sizeof(GraphElem)*nv, cuStreams[3]);

    GraphElem* newCommIds = new GraphElem[nv];
    GraphWeight* e_cj = new GraphWeight[max_order];

    cudaStreamSynchronize(cuStreams[0]);
    cudaStreamSynchronize(cuStreams[1]);
    cudaStreamSynchronize(cuStreams[2]);

    for(GraphElem v = v0; v < v1; ++v)
    {
        //loop through all neighboring clusters
        GraphElem unique_id = 0;
        std::unordered_map<GraphElem, GraphElem> neighCommIdMap;

        GraphWeight e_ci = 0.;
        GraphWeight ki = vertexWeights_host[v];
        GraphElem start = indices[v];
        GraphElem end = indices[v+1];
        GraphElem my_comm_id = commIds_host[v];

        for(GraphElem j = start; j < end; ++j)
        {
            GraphElem u = edges[j];
            GraphWeight w_vu = edgeWeights[j];

            GraphElem comm_id = commIds_host[u];
            if(comm_id != my_comm_id)
            {
                std::unordered_map<GraphElem,GraphElem>::iterator iter;
                if((iter = neighCommIdMap.find(comm_id)) == neighCommIdMap.end())
                {
                    neighCommIdMap.insert({comm_id, unique_id});
                    e_cj[unique_id] = w_vu; //graph->get_weight(vi,j);
                    unique_id++;
                }
                else
                    e_cj[iter->second] += w_vu;
            }
            else if(v != u)
                e_ci += w_vu;
        }

        //determine the best move
        GraphWeight ac_i = commWeights_host[my_comm_id]-ki;

        GraphWeight delta = 0;
        GraphElem destCommId = my_comm_id;

        for(auto iter = neighCommIdMap.begin(); iter != neighCommIdMap.end(); ++iter)
        {
            GraphWeight val = e_cj[iter->second];
            val -= (e_ci - ki*(ac_i-commWeights_host[iter->first])/(2.*mass));
            val /= mass;
            if(val > delta)
            {
                destCommId = iter->first;
                delta = val;
            }
        }
        newCommIds[v] = destCommId;
    }
    cudaStreamSynchronize(cuStreams[3]);

    GraphWeight err = 0;
    for(GraphElem i = v0; i < v1; ++i)
        err += std::pow(newCommIds_host[i]-newCommIds[i],2);
    if(err != 0)
        std::cout << "There are errors in the Louvain update.\n";// << err << std::endl;
    else
        std::cout << "The results from GPU Louvain update are the same as the CPU.\n";
    if(err != 0)
    {
        std::cout << "These are different predictions (GPU and CPU):\n";
        for(GraphElem i = v0; i < v1; ++i)
            if(newCommIds_host[i] != newCommIds[i])
                std::cout << i << " " << newCommIds_host[i] << " " << newCommIds[i] << std::endl;
    }
    delete [] e_cj;
    delete [] newCommIds;

    CudaFreeHost(commIds_host);
    CudaFreeHost(commWeights_host);
    CudaFreeHost(newCommIds_host);
    CudaFreeHost(vertexWeights_host);

    cudaStreamDestroy(cuStreams[0]);
    cudaStreamDestroy(cuStreams[1]);
    cudaStreamDestroy(cuStreams[2]); 
    cudaStreamCreate(&cuStreams[3]);
}

GraphWeight check_modularity
(
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices, 
    GraphElem* commIds_dev,
    GraphWeight* commWeights_dev,
    const GraphWeight& mass,
    const GraphElem& nv
)
{
    GraphElem* commIds = new GraphElem[nv];
    GraphWeight* commWeights = new GraphWeight[nv];

    CudaMemcpyAsyncDtoH(commIds, commIds_dev, sizeof(GraphElem)*nv,0);
    CudaMemcpyAsyncDtoH(commWeights, commWeights_dev, sizeof(GraphWeight)*nv ,0);
 
    GraphWeight mod = 0.;

    uint32_t* comm_id_list = new uint32_t[(nv+31)>>5];
    for(GraphElem  i = 0; i < (nv+31)>>5; ++i)
        comm_id_list[i] = 0;

    CudaDeviceSynchronize();

    for(GraphElem u = 0; u < nv; ++u)
    {
        GraphElem my_comm_id = commIds[u];
        GraphElem start, end;
        start = indices[u];
        end = indices[u+1];
        for(GraphElem j = start; j < end; ++j)
        {
            GraphElem v = edges[j];
            if(commIds[v] == my_comm_id)
                mod += edgeWeights[j];
        }
        uint32_t is_set = comm_id_list[my_comm_id>>5];
        uint32_t flag = 1<<(my_comm_id%32);
        is_set &= flag;

        if(!is_set)
        {
            comm_id_list[my_comm_id>>5] |= flag;
            GraphWeight ac = commWeights[my_comm_id];
            mod -= ac*ac/(2.*mass);
        }
    }
    mod /= (2.*mass);
    delete [] comm_id_list;
    delete [] commIds;
    delete [] commWeights;
    return mod;
}
