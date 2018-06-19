#ifndef __architecture__
#define __architecture__


#include <vector>
#include <iostream>
using namespace std;

class computationalNode;

struct architecture{
    int Nlayers;
    vector<int> layer_dimension;
    vector<vector<int> > layer_shape;

    int Nweights;
    vector<int> weight_dimension;
    vector<vector<int> > weight_shape;
    vector<int> bias_len;

    int Nnodes;
    vector<computationalNode*> computation_list;
    vector<int> from;
    vector<int> to;



    void SetInputShape(int dim1, int dim2 = -1, int dim3 = -1);
    void Add(computationalNode * node);
    void Print();

    architecture();
    ~architecture();
};

#endif // __architecture__
