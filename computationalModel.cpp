#include "computationalModel.h"
#include "layers.h"
#include "computationalNode.h"
#include <iostream>
#include "mathFunc.h"
#include <tuple>
using namespace std;

void computationalModel::AddNode(int from, int to, computationalNode* node){
    computationList.push_back(std::make_tuple(from, to, node));
}

void computationalModel::SetModel(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, bool primalWeightOwner){
    Nlayers=layersData->Nlayers;

    AddNode(0, 0, new StairsFullConvolution(0, 3, 1, 10));
    AddNode(0, 1, new LastAveragePooling(20));
    AddNode(1, 1, new StairsFullConvolution(1, 20, 1, 20));
    AddNode(1, 2, new LastAveragePooling(40));
    AddNode(2, 2, new StairsFullConvolution(2, 40, 1, 40));
    AddNode(2, 3, new LastAveragePooling(80));
    AddNode(3, 3, new StairsFullConvolution(3, 80, 1, 80));
    AddNode(3, 4, new LastAveragePooling(160));
    AddNode(4, 5, new FullyConnectedSoftMax(4));

    this->Compile(layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner);
}

void computationalModel::SetModel(layers* layersData, layers* deltas, weights* weightsData, weights* gradient,
                  activityLayers* layersActivity, bool primalWeightOwner, computationalModel * primalCM){
    this->SetModel(layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner);
    computationalNode * node;
    for(int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        if (node->NeedsUnification())
            node->Unify(get<2>(primalCM->computationList[j]));
    }
}

void computationalModel::Compile(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, bool primalWeightOwner){
    hasBottomWeightDependency.resize(Nlayers, 0);

    computationalNode * node;
    int from, to;
    for(int j=0; j<computationList.size(); ++j){
        from = get<0>(computationList[j]);
        to   = get<1>(computationList[j]);
        node = get<2>(computationList[j]);
        if (node->HasWeightsDependency() || hasBottomWeightDependency[from])
                    hasBottomWeightDependency[to]=1;
        node->Initiate(layersData, deltas, weightsData, gradient, layersActivity, from, to, primalWeightOwner);

    }
}

//assuming input is set
void computationalModel::ForwardPass(){
    computationalNode* node;
    for(int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        node->ForwardPass();
    }
//    for(vector<pair<int, int> >::iterator it=computationList.begin(); it!=computationList.end(); ++it){
//        //cout<<it->first<<" "<<it->second<<endl;
//        computationTable[it->first][it->second]->ForwardPass();
//    }
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

//    vector<pair<int, int> >::reverse_iterator rit=computationList.rbegin();
//    for( ; rit!=computationList.rend(); ++rit){
//        computationTable[rit->first][rit->second]->BackwardPass(hasBottomWeightDependency[rit->first]);
//    }
}

void computationalModel::WriteCoefficientsToFile(){
    computationalNode* node;
    for(int j=0; j<computationList.size(); ++j){
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
    for(int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        node->SetToTrainingMode();
    }
}

void computationalModel::SetToTestMode(){
    computationalNode* node;
    for(int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        node->SetToTestMode();
    }
}

computationalModel::~computationalModel(){
    computationalNode* node;
    for(int j=0; j<computationList.size(); ++j){
        node = get<2>(computationList[j]);
        delete node;
    }
}
