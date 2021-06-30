#ifndef GRAPH_GPU_HPP_
#define GRAPH_GPU_HPP_
#include "types.hpp"
#include "graph.hpp"
class GraphGPU
{
  private:

    Graph* graph_;

    GraphElem nv_, ne_, nv_per_batch_;

    void *edges_;
    void *edgeWeights_;
    GraphElem2 *commIdKeys_;

    GraphElem* indices_;
    GraphWeight* vertexWeights_; 
    GraphElem* commIds_;
    GraphWeight* commWeights_;

    GraphElem maxOrder_;

    GraphElem* indicesHost_;
    GraphElem* edgesHost_;
    GraphWeight* edgeWeightsHost_;
    GraphElem*  indexOrdersHost_;

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
    void scan_edges();
    void scan_edge_weights();
    void sum_vertex_weights();
    void max_vertex_weights();


    GraphWeight* get_vertex_weights() { return vertexWeights_;};
    
};
#endif
