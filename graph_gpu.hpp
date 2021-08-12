#ifndef GRAPH_GPU_HPP_
#define GRAPH_GPU_HPP_
#include <vector>
#include "types.hpp"
#include "graph.hpp"
class GraphGPU
{
  private:

    Graph* graph_;

    GraphElem nv_, ne_, nv_per_batch_, ne_per_batch_;

    void *edges_;
    void *edgeWeights_;
    GraphElem2 *commIdKeys_;
    GraphElem* indexOrders_; 

    GraphElem* indices_;
    GraphWeight* vertexWeights_; 
    GraphElem* commIds_;
    GraphWeight* commWeights_;
    GraphElem* newCommIds_;
    GraphElem maxOrder_;
    GraphWeight mass_;

    GraphElem* indicesHost_;
    GraphElem* edgesHost_;
    GraphWeight* edgeWeightsHost_;
    //GraphElem*  indexOrdersHost_;
 
    std::vector<GraphElem> vertex_partition_;

  public:
    GraphGPU
    (
        Graph* graph 
    );
    ~GraphGPU();
    GraphElem determine_optimal_vertices_per_batch
    (
        const GraphElem& nv, 
        const GraphElem& ne, 
        const GraphElem& maxOrder, 
        const unsigned& unit_size
    );
    void set_communtiy_ids
    (
        GraphElem* commIds
    );
    void compute_community_weights();
    void sort_edges_by_community_ids();
    void singleton_partition();
    GraphElem max_order();
    float scan_edges();
    float scan_edge_weights();
    float sum_vertex_weights();
    float max_vertex_weights();

    void build_local_comm_offsets();

    GraphElem determine_optimal_edges_per_batch (const GraphElem&, const GraphElem&, const unsigned&);
    void determine_optimal_vertex_partition(GraphElem*, const GraphElem&, const GraphElem&, const GraphElem&, std::vector<GraphElem>&);

    GraphWeight* get_vertex_weights() { return vertexWeights_;};
    void louvain_update();
    void compute_mass();
    GraphWeight compute_modularity();

    void move_edges_to_device(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream = 0);
    void move_edges_to_host(const GraphElem& v0,  const GraphElem& v1, cudaStream_t stream = 0);
    void move_weights_to_device(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream = 0);
    void move_weights_to_host(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream = 0);

    void memcpy();
};
#endif
