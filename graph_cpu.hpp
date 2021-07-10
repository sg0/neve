#ifndef GRAPH_CPU_HPP_
#define GRAPH_CPU_HPP_
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
);

void check_louvain_update
(
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    GraphWeight* vertexWeights,
    GraphElem*   commIds_dev,
    GraphWeight* commWeights_dev,
    GraphElem*   newCommids_dev,
    const GraphWeight& mass,
    const GraphElem& max_order,
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& nv
);

GraphWeight check_modularity
(
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices, 
    GraphElem* commIds_dev,
    GraphWeight* commWeights_dev,
    const GraphWeight& mass,
    const GraphElem& nv
);

#endif
