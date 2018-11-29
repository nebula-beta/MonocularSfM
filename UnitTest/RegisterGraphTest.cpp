#include "Reconstruction/RegisterGraph.h"

#include <iostream>
#include <cassert>
using namespace MonocularSfM;
int main()
{

    RegisterGraph register_graph(5);
    register_graph.AddEdge(0, 1);
    register_graph.AddEdge(0, 2);
    register_graph.AddEdge(1, 2);
    register_graph.AddEdge(1, 3);
    register_graph.AddEdge(2, 4);

    register_graph.SetRegistered(0);
    register_graph.SetRegistered(1);
    std::vector<image_t> ids = register_graph.GetNextImageIds();

    assert(ids.size() == 2);
    assert(ids[0] == 2);
    assert(ids[1] == 3);

    register_graph.SetRegistered(2);

    ids = register_graph.GetNextImageIds();

    assert(ids.size() == 2);

    std::cout << "Test RegisterGraph SUCCESS!!!" << std::endl;

    return 0;
}
