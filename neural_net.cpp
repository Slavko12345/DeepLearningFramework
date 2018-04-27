#include "neural_net.h"
#include "mathFunc.h"
#include "time.h"
#include <math.h>
#include "vect.h"
#include "computationalNode.h"
#include "layers.h"
#include "weights.h"
#include "computationalModel.h"
#include "data.h"
#include "activityLayers.h"
#include <stdlib.h>
#include <iostream>
#include "activityData.h"
#include "globals.h"
#include "tensor.h"
using namespace std;

void NeuralNet::Initiate()
{
    testMode = 1;
    primalWeightOwner = 1;

    layersData=new layers;
    layersData->SetModel();
    Nlayers=layersData->Nlayers;

    deltas=new layers;
    deltas->SetModel();

    layersActivity = new activityLayers;
    layersActivity->SetModel(layersData);
    layersActivity->SetAllActive();
    cout<<"Dropping: "<<layersActivity->dropping<<endl;

    weightsData=new weights;
    weightsData->SetModel();

    SetRandomWeights(MAX_ABS_RANDOM_WEIGHTS);
    Nweights=weightsData->Nweights;

    gradient=new weights;
    gradient->SetModel();

    computation=new computationalModel;
    computation->SetModel(layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner);


}

void NeuralNet::Initiate(NeuralNet * NN){
    testMode = 1;
    primalWeightOwner = 0;

    layersData=new layers;
    layersData->SetModel();
    Nlayers=layersData->Nlayers;

    deltas=new layers;
    deltas->SetModel();

    layersActivity = new activityLayers;
    layersActivity->SetModel(layersData);
    layersActivity->SetAllActive();

    weightsData = NN->weightsData;
    Nweights = NN->weightsData->Nweights;

    gradient=new weights;
    gradient->SetModel();

    computation=new computationalModel;
    computation->SetModel(layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner, NN->computation);
}




void NeuralNet::SetRandomWeights(double bound){
    srand(time(NULL));
    weightsData->SetToRandomValues(bound);
}


void NeuralNet::ForwardPass(orderedData* input){
    layersData->SetInnerLayersToZero();
    layersData->SetInput(input, layersActivity, testMode);
    computation->ForwardPass();
//    cout<<"All layers: "<<endl;
//    for(int j=0; j<Nlayers; ++j){
//        cout<<"layer "<<j<<": "<<endl;
//        this->layersData->layerList[j]->Print();
//        cout<<endl;
//    }
//    int k;
//    cin>>k;
//    cout<<endl;
}

void NeuralNet::BackwardPass(int trueClass, int input_len){
    //deltas->SetInnerLayersToZero();
    deltas->SetLayersToZero();
    //deltas->layerList[0]->SetToZeroStartingFrom(input_len);
    //deltas->SetOutputDelta(layersData, trueClass);
    computation->BackwardPass(trueClass);
}

void NeuralNet::ForwardBackwardPass(orderedData* input, int trueClass){
    ForwardPass(input);
    if (FOCUSED_TRAINING){
        if (layersData->layerList[Nlayers-1]->elem[trueClass] < FOCUSED_PROBABILITY_THRESHOLD)
            BackwardPass(trueClass, input->len);
    }

    else
        BackwardPass(trueClass, input->len);
}

void NeuralNet::CalculateGradient(Data* inputData){
    gradient->SetToZero();
    tensor* data_j = new tensor();
    if (layersActivity->dropping)
        for(int j=0; j<inputData->totalSize(); ++j){
            //layersActivity->SetAllActive();
            layersActivity->DropUnits();
            data_j->SetToTLayer(inputData->classData, j);
            ForwardBackwardPass(data_j, inputData->labels[j]);
        }

    else{
        for(int j=0; j<inputData->totalSize(); ++j){
            data_j->SetToTLayer(inputData->classData, j);
            ForwardBackwardPass(data_j, inputData->labels[j]);
        }
    }

    DeleteOnlyShell(data_j);
}

void NeuralNet::CalculateGradientFunctionValue(Data* inputData, double& functionVal){
    tensor* data_j = new tensor();
    gradient->SetToZero();
    functionVal = 0;
    orderedData* lastLayerLink = layersData->layerList[Nlayers-1];
    int total = inputData->totalSize(), lab;
    if (layersActivity->dropping)
        for(int j=0; j<total; ++j){
            //layersActivity->SetAllActive();
            layersActivity->DropUnits();
            data_j->SetToTLayer(inputData->classData, j);
            lab = inputData->labels[j];
            ForwardBackwardPass(data_j, lab);
            functionVal-=log(lastLayerLink->elem[lab]);
            ++total;
        }

    else{
        for(int j=0; j<total; ++j){
            lab = inputData->labels[j];
            data_j->SetToTLayer(inputData->classData, j);
            ForwardBackwardPass(data_j, lab);
            functionVal-=log(lastLayerLink->elem[lab]);
            ++total;
        }
    }
    functionVal/=double(total);
    DeleteOnlyShell(data_j);
}


void NeuralNet::CalculateErrorAndAccuracy(Data* inputData, double &error, double &accuracy){
    tensor* data_j = new tensor();
    error=0;
    int correct=0, predictedLabel, lab;
    int total = inputData->totalSize();
    orderedData* lastLayerLink = layersData->layerList[Nlayers-1];
    if (NEW_ERROR_FUNCTION)
        for(int j=0; j<total; ++j){
            data_j->SetToTLayer(inputData->classData, j);
            ForwardPass(data_j);
            predictedLabel=lastLayerLink->ArgMax();
            lab = inputData->labels[j];
            if (predictedLabel==lab) correct++;
            error -= lastLayerLink->elem[lab];
        }
    else
        for(int j=0; j<total; ++j){
            data_j->SetToTLayer(inputData->classData, j);
            ForwardPass(data_j);
            predictedLabel=lastLayerLink->ArgMax();
            lab = inputData->labels[j];
            if (predictedLabel==lab) correct++;
            error-=log(fabs(lastLayerLink->elem[lab])+1E-10);
        }
    error/=double(total);
    accuracy = double(correct)/total;

    DeleteOnlyShell(data_j);
}

void NeuralNet::CalculateSubErrorAndAccuracy(Data* inputData, double &error, int &correct){
    tensor* data_j = new tensor();
    error=0; correct=0;
    int predictedLabel, lab;
    int total = inputData->totalSize();
    orderedData* lastLayerLink = layersData->layerList[Nlayers-1];
    if (NEW_ERROR_FUNCTION)
        for(int j=0; j<total; ++j){
            data_j->SetToTLayer(inputData->classData, j);
            ForwardPass(data_j);
            predictedLabel=lastLayerLink->ArgMax();
            lab = inputData->labels[j];
            if (predictedLabel==lab) correct++;
            error -= lastLayerLink->elem[lab];
        }
    else
        for(int j=0; j<total; ++j){
            data_j->SetToTLayer(inputData->classData, j);
            ForwardPass(data_j);
            predictedLabel=lastLayerLink->ArgMax();
            lab = inputData->labels[j];
            if (predictedLabel==lab) correct++;
            error-=log(fabs(lastLayerLink->elem[lab])+1E-10);
        }

    DeleteOnlyShell(data_j);
}

double NeuralNet::CalculateAccuracy(Data* inputData){
    tensor* data_j = new tensor();
    int correct=0, total = inputData->totalSize(), predictedLabel, lab;
    orderedData* lastLayerLink =layersData->layerList[Nlayers-1];
    for(int j=0; j<total; ++j){
        data_j->SetToTLayer(inputData->classData, j);
        ForwardPass(data_j);
        predictedLabel=lastLayerLink->ArgMax();
        lab = inputData->labels[j];
        if (predictedLabel==lab) correct++;
    }
    DeleteOnlyShell(data_j);

    return double(correct)/total;
}

void NeuralNet::PrintProbabilities(){
    layersData->layerList[Nlayers-1]->Print();
}

void NeuralNet::SwitchToTrainingMode(){
    if (!testMode){
        cout<<"Already in Training Mode"<<endl;
        return;
    }

    testMode=0;
    computation->SetToTrainingMode();
}

void NeuralNet::SwitchToTestMode(){
    if (testMode){
        cout<<"Already in Test Mode"<<endl;
        return;
    }
    testMode=1;

    layersActivity->SetAllActive();
    computation->SetToTestMode();
}

NeuralNet::~NeuralNet(){
    delete layersData;
    delete deltas;
    delete layersActivity;
    if (primalWeightOwner)
        delete weightsData;
    delete gradient;
    delete computation;
}

