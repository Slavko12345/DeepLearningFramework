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
#include <float.h>
#include "architecture.h"
#include "tensor4D.h"
using namespace std;

NeuralNet::NeuralNet(){
    arch = new architecture();
    architectureBased = false;
}

void NeuralNet::Initiate()
{
    if (architectureBased){
        InitiateFromArchitecture();
        return;
    }

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

void NeuralNet::InitiateFromArchitecture(){

    if (!arch->input_shape_set)
        cout<<"Error: input shape is not set for architecture"<<endl;

    if (!arch->num_classes_set)
        cout<<"Error: number of classes is not set for architecture"<<endl;

    testMode = 1;
    primalWeightOwner = 1;

    layersData=new layers;
    layersData->SetModel(arch);
    Nlayers=layersData->Nlayers;

    deltas=new layers;
    deltas->SetModel(arch);

    layersActivity = new activityLayers;
    layersActivity->SetModel(layersData);
    layersActivity->SetAllActive();
    cout<<"Dropping: "<<layersActivity->dropping<<endl;

    weightsData=new weights;
    weightsData->SetModel(arch);

    SetRandomWeights(MAX_ABS_RANDOM_WEIGHTS);
    Nweights=weightsData->Nweights;

    gradient=new weights;
    gradient->SetModel(arch);

    computation=new computationalModel;
    computation->SetModel(arch, layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner);

    cout<<"Architecture: "<<endl;
    arch->Print();
    cout<<endl;
}

void NeuralNet::Initiate(NeuralNet * NN){
    if (NN->architectureBased){
        InitiateFromArchitecture(NN);
        return;
    }

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

void NeuralNet::InitiateFromArchitecture(NeuralNet * NN){
    testMode = 1;
    primalWeightOwner = 0;

    delete arch;
    arch = NN->arch;
    architectureBased = true;

    layersData=new layers;
    layersData->SetModel(arch);
    Nlayers=layersData->Nlayers;

    deltas=new layers;
    deltas->SetModel(arch);

    layersActivity = new activityLayers;
    layersActivity->SetModel(layersData);
    layersActivity->SetAllActive();

    weightsData = NN->weightsData;
    Nweights = NN->weightsData->Nweights;

    gradient=new weights;
    gradient->SetModel(arch);

    computation=new computationalModel;
    computation->SetModel(arch, layersData, deltas, weightsData, gradient, layersActivity, primalWeightOwner, NN->computation);
}

void NeuralNet::SetInputShape(int dim1, int dim2, int dim3){
    arch->SetInputShape(dim1, dim2, dim3);
    architectureBased = true;
}

void NeuralNet::Add(computationalNode * node){
    arch->Add(node);
    architectureBased = true;
}




void NeuralNet::SetRandomWeights(float bound){
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
    if (testMode)
        this->SwitchToTrainingMode();

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

void NeuralNet::CalculateGradientFunctionValue(Data* inputData, float& functionVal){
    cout<<"Deprecated function CalculateGradientFunctionValue"<<endl;
    if (!testMode)
        this->SwitchToTestMode();

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
            functionVal-=log(lastLayerLink->elem[lab] + FLT_EPSILON);
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
    functionVal/=float(total);
    DeleteOnlyShell(data_j);
}


void NeuralNet::CalculateErrorAndAccuracy(Data* inputData, float &error, float &accuracy){
    if (!testMode)
        this->SwitchToTestMode();

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
            error-=log(fabs(lastLayerLink->elem[lab]) + FLT_EPSILON);
        }
    error/=float(total);
    accuracy = float(correct)/total;

    DeleteOnlyShell(data_j);
}

void NeuralNet::CalculateSubErrorAndAccuracy(Data* inputData, float &error, int &correct){
    if (!testMode)
        this->SwitchToTestMode();

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
            error-=log(fabs(lastLayerLink->elem[lab]) + FLT_EPSILON);
        }

    DeleteOnlyShell(data_j);
}

float NeuralNet::CalculateAccuracy(Data* inputData){
    if (!testMode)
        this->SwitchToTestMode();
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

    return float(correct)/total;
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

void NeuralNet::UpdateBalancedDropParameters(float alpha_, float pDrop_, float pNotDrop_){
    computation->UpdateBalancedDropParameters(alpha_, pDrop_, pNotDrop_);
}

bool NeuralNet::CheckCompatibility(Data * data){
    if (arch->num_classes != data->C){
        cout<<"Error: Number of target classes is not equal!"<<endl;
        return false;
    }

    if (arch->input_dimension != 3 || arch->input_shape[0] != data->classData->depth ||
        arch->input_shape[1] != data->classData->rows || arch->input_shape[2] != data->classData->cols){
            cout<<"Error: Input shape is not the same!"<<endl;
            return false;
    }

    return true;

}

NeuralNet::~NeuralNet(){
    delete layersData;
    delete deltas;
    delete layersActivity;
    if (primalWeightOwner){
        delete weightsData;
        delete arch;
    }

    delete gradient;
    delete computation;
}

