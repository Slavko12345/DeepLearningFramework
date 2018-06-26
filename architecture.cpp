#include "architecture.h"
#include "computationalNode.h"

architecture::architecture(){
    Nlayers = 0;
    layer_dimension.clear();
    layer_shape.clear();

    Nweights = 0;
    weight_dimension.clear();
    weight_shape.clear();
    bias_len.clear();

    Nnodes = 0;
    computation_list.clear();
    from.clear();
    to.clear();

    input_shape_set = false;
    num_classes_set = false;
}

void architecture::SetInputShape(int dim1, int dim2, int dim3){
    input_dimension = 3;
    if (dim3 == -1) input_dimension = 2;
    if (dim2 == -1) input_dimension = 1;

    layer_dimension.push_back(input_dimension);
    input_shape = {dim1, dim2, dim3};
    layer_shape.push_back(input_shape);
    ++Nlayers;

    input_shape_set = true;
}

void architecture::Add(computationalNode * node){
    computation_list.push_back(node);
    ++Nnodes;
    node->UpdateArchitecture(this);
}

void architecture::Print(){
    cout<<"Layers: "<<endl;
    for(int j=0; j<Nlayers; ++j){
        cout<<"Layer "<<j<<": ";
        if (layer_dimension[j] == 1)
            cout<<"vector: \t";
        if (layer_dimension[j] == 2)
            cout<<"matrix: \t";
        if (layer_dimension[j] == 3)
            cout<<"tensor: \t";
        for(int k = 0; k<layer_dimension[j]; ++k)
            cout<<layer_shape[j][k]<<"\t";
        cout<<endl;
    }

    int weights_num, total_weight_num = 0;

    cout<<endl<<"Weights: "<<endl;
    for(int j=0; j<Nweights; ++j){
        cout<<"Weight #"<<j<<": ";
        if (weight_dimension[j] == 1)
            cout<<"vector: \t";
        if (weight_dimension[j] == 2)
            cout<<"matrix: \t";
        if (weight_dimension[j] == 3)
            cout<<"tensor: \t";

        weights_num = 1;
        for(int k = 0; k<weight_dimension[j]; ++k){
            cout<<weight_shape[j][k]<<"\t";
            weights_num *= weight_shape[j][k];
        }

        cout<<"\t Bias: vector: "<<bias_len[j]<<"\t len: "<<weights_num + bias_len[j]<<endl;
        total_weight_num += weights_num + bias_len[j];
    }
    cout<<"Total number of coefficients: "<<total_weight_num<<endl;



    cout<<endl<<"Nodes: "<<endl;
    for(int j=0; j<Nnodes; ++j){
        cout<<"Computation from "<<from[j]<<" to: "<<to[j]<<": "<<endl;
        computation_list[j]->Print();
        cout<<endl;
    }
}

architecture::~architecture(){
    for(int j=0; j<Nnodes; ++j)
        delete computation_list[j];
}
