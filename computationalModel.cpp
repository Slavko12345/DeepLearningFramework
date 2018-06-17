#include "computationalModel.h"
#include "layers.h"
#include "computationalNode.h"
#include <iostream>
#include "mathFunc.h"
#include <tuple>
#include "orderedData.h"
#include "weights.h"
using namespace std;

void computationalModel::AddNode(int from, int to, computationalNode* node){
    computationList.push_back(std::make_tuple(from, to, node));
}

void computationalModel::SetModel(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, bool primalWeightOwner){
    Nlayers=layersData->Nlayers;
    nnlayers = layersData;
    nndeltas = deltas;
    nnweights = weightsData;
    nngrad = gradient;

    //AddNode(0, 0, new InputBalancedDrop());
    AddNode(0, 0, new StairsFullBottleneck(0, 1, 3,  4, 4, 8));
    AddNode(0, 1, new FullAveragePooling());
    AddNode(1, 1, new StairsFullBottleneck(2, 3, 35, 4, 4, 8));
    AddNode(1, 2, new FullAveragePooling());
    AddNode(2, 2, new StairsFullBottleneck(4, 5, 67, 4, 4, 8));
    AddNode(2, 3, new FullAveragePooling());
    AddNode(3, 4, new FullyConnectedSoftMax(6));

    this->Compile(layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner);
}

void computationalModel::SetModel(layers* layersData, layers* deltas, weights* weightsData, weights* gradient,
                  activityLayers* layersActivity, bool primalWeightOwner, computationalModel * primalCM){
    this->SetModel(layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner);
    computationalNode * node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        if (node->NeedsUnification())
            node->Unify(get<2>(primalCM->computationList[j]));
    }
}

void computationalModel::UpdateBalancedDropParameters(float alpha_, float pDrop_, float pNotDrop_){
    computationalNode * node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        if (node->UsesBalancedDrop())
            node->UpdateBalancedDropParameters(alpha_, pDrop_, pNotDrop_);
    }
}

void computationalModel::Compile(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, bool primalWeightOwner){
    hasBottomWeightDependency.resize(Nlayers, 0);

    computationalNode * node;
    int from, to;
    for(unsigned int j=0; j<computationList.size(); ++j){
        from = get<0>(computationList[j]);
        to   = get<1>(computationList[j]);
        node = get<2>(computationList[j]);
        if (node->HasWeightsDependency() || hasBottomWeightDependency[from])
                    hasBottomWeightDependency[to] = 1;
        node->Initiate(layersData, deltas, weightsData, gradient, layersActivity, from, to, primalWeightOwner);

    }
}

//assuming input is set
void computationalModel::ForwardPass(){
    computationalNode* node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        node->ForwardPass();
    }

}

//assuming last layer deltas are set
void computationalModel::BackwardPass(int trueClass){
    computationalNode* node;
    int from;
    for(int j=computationList.size()-1; j>=0; --j){
        from = get<0>(computationList[j]);
        node = get<2>(computationList[j]);
        node->BackwardPass(hasBottomWeightDependency[from], trueClass);
    }
}

void computationalModel::WriteCoefficientsToFile(){
    computationalNode* node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        if (node->HasWeightsDependency())
            node->WriteStructuredWeightsToFile();
    }
//    for(vector<pair<int, int> >::iterator it=computationList.begin(); it!=computationList.end(); ++it){
//        if (computationTable[it->first][it->second]->HasWeightsDependency())
//            computationTable[it->first][it->second]->WriteStructuredWeightsToFile();
//    }
}

void computationalModel::SetToTrainingMode(){
    computationalNode* node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        node->SetToTrainingMode();
    }
}

void computationalModel::SetToTestMode(){
    computationalNode* node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        node->SetToTestMode();
    }
}

computationalModel::~computationalModel(){
    computationalNode* node;
    for(unsigned int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        delete node;
    }
}
