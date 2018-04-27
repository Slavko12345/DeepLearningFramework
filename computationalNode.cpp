#include "computationalNode.h"

#include <iostream>
#include "layers.h"
#include "weights.h"
#include "vect.h"
#include "matrix.h"
#include "realNumber.h"
#include "tensor.h"
#include "tensor4D.h"
#include "orderedData.h"
#include "mathFunc.h"
#include <math.h>
#include "activityLayers.h"
#include "activityData.h"
#include "globals.h"
using namespace std;

computationalNode::computationalNode(){
    testMode = 1;
}

void computationalNode::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    cout<<"Error: Call of Initiate for abstract class"<<endl;
}

void computationalNode::SetToTrainingMode(){
    cout<<"Error: Call of SetToTrainingMode for abstract class"<<endl;
}

void computationalNode::SetToTestMode(){
    cout<<"Error: Call of SetToTestMode for abstract class"<<endl;
}

bool computationalNode::NeedsUnification(){
    return 0;
}

void computationalNode::Unify(computationalNode * primalCN){
    cout<<"Error: call of Unify for abstract class"<<endl;
}

void computationalNode::WriteStructuredWeightsToFile(){
    cout<<"Error: call of WriteStructuredWeightsToFile for abstract class"<<endl;
}

computationalNode::~computationalNode(){
}




FullyConnected::FullyConnected(int weightsNum_): weightsNum(weightsNum_){
}

void FullyConnected::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=layersData->layerList[from];
    output=layersData->layerList[to];

    inputDelta = deltas->layerList[from];
    outputDelta = deltas->layerList[to];

    kernel=    static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(   gradient->weightList[weightsNum].dataWeight);

    bias=    static_cast<vect*>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect*>(   gradient->weightList[weightsNum].bias);

    indexInput.reserve(kernel->cols);
    indexOutput.reserve(kernel->rows);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->len != kernel->cols || output->len != kernel->rows || bias->len != output->len)
        cout<<"Error in Fully Connected from layer "<<from<<" to layer "<<to<<endl;
}


void FullyConnected::ForwardPass(){
//    input->ListNonzeroActiveElements(indexInput, inputActivity);
//    output->ListActiveElements(indexOutput, outputActivity);
//
//    output->AddMatrVectProductBias(kernel, input, bias, indexInput, indexOutput);

    output->AddMatrVectProductBias(kernel, input, bias);
    output->SetDroppedElementsToZero(outputActivity);
}

void FullyConnected::BackwardPass(bool computeDelta, int trueClass){
//    outputDelta->ListNonzeroActiveElements(indexOutput, outputActivity);
//    if (computeDelta){
//        inputDelta->BackwardFullyConnected(kernel, outputDelta, input, kernelGrad, biasGrad, indexInput, indexOutput);
//    }
//
//    else
//        kernelGrad->BackwardFullyConnectedOnlyGrad(outputDelta, input, biasGrad, indexOutput, indexInput);
//        //kernelGrad->AddOuterProduct(outputDelta, input, biasGrad, indexOutput, indexInput);

    if (computeDelta){
        inputDelta->BackwardFullyConnected(kernel, outputDelta, input, kernelGrad, biasGrad);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }
    else
        kernelGrad->BackwardFullyConnectedOnlyGrad(outputDelta, input, biasGrad);
}

bool FullyConnected::HasWeightsDependency(){
    return 1;
}

FullyConnected::~FullyConnected(){
}





FullyConnectedSoftMax::FullyConnectedSoftMax(int weightsNum_): weightsNum(weightsNum_){
}

void FullyConnectedSoftMax::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=layersData->layerList[from];
    output=layersData->layerList[to];

    inputDelta = deltas->layerList[from];
    outputDelta = deltas->layerList[to];

    kernel=    static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(   gradient->weightList[weightsNum].dataWeight);

    bias=    static_cast<vect*>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect*>(   gradient->weightList[weightsNum].bias);

    //indexInput.reserve(input->len);
    //compressedInput = new vect(input->len);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->len != kernel->cols || output->len != kernel->rows || bias->len != output->len)
        cout<<"Error in Fully Connected from layer "<<from<<" to layer "<<to<<endl;
}


void FullyConnectedSoftMax::ForwardPass(){
//    input->ListNonzeroActiveElements(indexInput, inputActivity, compressedInput);
//    output->AddMatrCompressedVectProductBias(kernel, compressedInput, bias, indexInput);
    output->AddMatrVectProductBias(kernel, input, bias);
    SoftMax(output, output);
}

void FullyConnectedSoftMax::BackwardPass(bool computeDelta, int trueClass){
    outputDelta->Copy(output);
    outputDelta->elem[trueClass] -= 1.0;
    if (computeDelta){
        //inputDelta->BackwardCompressedInputFullyConnected(kernel, outputDelta, compressedInput, kernelGrad, biasGrad, indexInput);
        inputDelta->BackwardFullyConnected(kernel, outputDelta, input, kernelGrad, biasGrad);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }
    else
        kernelGrad->BackwardFullyConnectedOnlyGrad(outputDelta, input, biasGrad);
        //kernelGrad->BackwardCompressedInputFullyConnectedOnlyGrad(outputDelta, compressedInput, biasGrad, indexInput);
}


void FullyConnectedSoftMax::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void FullyConnectedSoftMax::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode=1;
}

bool FullyConnectedSoftMax::HasWeightsDependency(){
    return 1;
}

void FullyConnectedSoftMax::WriteStructuredWeightsToFile(){
    char fileName[] = STRUCTERED_WEIGHTS;
    fileName[9] = '0' + weightsNum;
    ofstream f(fileName);
    f<<"bias: "<<endl;
    for(int j=0; j<bias->len; ++j)
        f<<bias->elem[j]<<endl;
    f<<endl<<"kernel: "<<endl;
    for(int c=0; c<kernel->cols; ++c){
        for(int r=0; r<kernel->rows; ++r)
            f<<kernel->At(r,c)<<'\t';
        f<<endl;
    }
    f.close();
}


FullyConnectedSoftMax::~FullyConnectedSoftMax(){
    //delete compressedInput;
}





FullyConnectedNoBiasSoftMax::FullyConnectedNoBiasSoftMax(int weightsNum_): weightsNum(weightsNum_){
}

void FullyConnectedNoBiasSoftMax::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=layersData->layerList[from];
    output=layersData->layerList[to];

    inputDelta = deltas->layerList[from];
    outputDelta = deltas->layerList[to];

    kernel=    static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(   gradient->weightList[weightsNum].dataWeight);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->len != kernel->cols || output->len != kernel->rows)
        cout<<"Error in Fully Connected from layer "<<from<<" to layer "<<to<<endl;
}


void FullyConnectedNoBiasSoftMax::ForwardPass(){
//    input->ListNonzeroActiveElements(indexInput, inputActivity, compressedInput);
//    output->AddMatrCompressedVectProductBias(kernel, compressedInput, bias, indexInput);
    output->AddMatrVectProduct(kernel, input);
    SoftMax(output, output);
}

void FullyConnectedNoBiasSoftMax::BackwardPass(bool computeDelta, int trueClass){
    outputDelta->Copy(output);
    outputDelta->elem[trueClass] -= 1.0;
    if (computeDelta){
        //inputDelta->BackwardCompressedInputFullyConnected(kernel, outputDelta, compressedInput, kernelGrad, biasGrad, indexInput);
        inputDelta->BackwardFullyConnectedNoBias(kernel, outputDelta, input, kernelGrad);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }
    else
        kernelGrad->BackwardFullyConnectedNoBiasOnlyGrad(outputDelta, input);
        //kernelGrad->BackwardCompressedInputFullyConnectedOnlyGrad(outputDelta, compressedInput, biasGrad, indexInput);
}


void FullyConnectedNoBiasSoftMax::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void FullyConnectedNoBiasSoftMax::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode=1;
}

bool FullyConnectedNoBiasSoftMax::HasWeightsDependency(){
    return 1;
}

void FullyConnectedNoBiasSoftMax::WriteStructuredWeightsToFile(){
    char fileName[] = STRUCTERED_WEIGHTS;
    fileName[9] = '0' + weightsNum;
    ofstream f(fileName);
    f<<endl<<"kernel: "<<endl;
    for(int c=0; c<kernel->cols; ++c){
        for(int r=0; r<kernel->rows; ++r)
            f<<kernel->At(r,c)<<'\t';
        f<<endl;
    }
    f.close();
}


FullyConnectedNoBiasSoftMax::~FullyConnectedNoBiasSoftMax(){
    //delete compressedInput;
}






Ensemble::Ensemble(int weightsNum_, int lastLayers_): weightsNum(weightsNum_), lastLayers(lastLayers_){
}

void Ensemble::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input = static_cast<tensor*>(layersData->layerList[from]);
    output = layersData->layerList[to];

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);

    kernel=    static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(   gradient->weightList[weightsNum].dataWeight);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialInput = new tensor();
    partialInput->SubLastTensor(input, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubLastTensor(inputDelta, lastLayers);

    separateInput = new tensor(input->rows, input->cols, lastLayers);
    separateInputDelta = new tensor(input->rows, input->cols, lastLayers);

    separateOutput = new tensor(input->rows, input->cols, output->len);
    separateOutputDelta = new tensor(input->rows, input->cols, output->len);

    pooledInput = new tensor(lastLayers, 1, 1);

    separateInput_r = new matrix();
    separateOutput_r = new matrix();
    separateInputDelta_r = new matrix();
    separateOutputDelta_r = new matrix();
    separateInput_rc = new vect();
    separateOutput_rc = new vect();
    separateInputDelta_rc = new vect();
    separateOutputDelta_rc = new vect();

    if (lastLayers != kernel->cols ||
        output->len != kernel->rows ||
        lastLayers > input->depth)
        cout<<"Error in Ensemble from layer "<<from<<" to layer "<<to<<endl;
}

void Ensemble::ForwardPass(){
    if (testMode){
        AveragePool3D_all(partialInput, pooledInput);
        output->AddMatrVectProduct(kernel, pooledInput);
        SoftMax(output, output);
    }

    if (!testMode){
        separateInput->Rearrange(partialInput);
        separateOutput->SetToZero();
        for(int r=0; r<input->rows; ++r){
            separateInput_r->SetToTensorLayer(separateInput, r);
            separateOutput_r->SetToTensorLayer(separateOutput, r);
            for(int c=0; c<input->cols; ++c){
                separateInput_rc->SetToMatrixRow(separateInput_r, c);
                separateOutput_rc->SetToMatrixRow(separateOutput_r, c);
                separateOutput_rc->AddMatrVectProduct(kernel, separateInput_rc);
                SoftMax(separateOutput_rc, separateOutput_rc);
            }
        }

    }
}

void Ensemble::BackwardPass(bool computeDelta, int trueClass){
    separateOutputDelta->Copy(separateOutput);
    for(int r=0; r<input->rows; ++r)
        for(int c=0; c<input->cols; ++c)
            separateOutputDelta->At(r, c, trueClass) -= 1.0;
    separateOutputDelta->Multiply(1.0 / (input->rows * input->cols));

    separateInputDelta->SetToZero();
    for(int r=0; r<input->rows; ++r){
        separateInput_r->SetToTensorLayer(separateInput, r);
        separateInputDelta_r->SetToTensorLayer(separateInputDelta, r);
        separateOutputDelta_r->SetToTensorLayer(separateOutputDelta, r);
        for(int c=0; c<input->cols; ++c){
            separateInput_rc->SetToMatrixRow(separateInput_r, c);
            separateInputDelta_rc->SetToMatrixRow(separateInputDelta_r, c);
            separateOutputDelta_rc->SetToMatrixRow(separateOutputDelta_r, c);
            separateInputDelta_rc->BackwardFullyConnectedNoBias(kernel, separateOutputDelta_rc, separateInput_rc, kernelGrad);
        }
    }
    partialInputDelta->BackwardRearrange(separateInputDelta);

    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void Ensemble::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void Ensemble::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode=1;
}

bool Ensemble::HasWeightsDependency(){
    return 1;
}

void Ensemble::WriteStructuredWeightsToFile(){
    char fileName[] = STRUCTERED_WEIGHTS;
    fileName[9] = '0' + weightsNum;
    ofstream f(fileName);
    f<<endl<<"kernel: "<<endl;
    for(int c=0; c<kernel->cols; ++c){
        for(int r=0; r<kernel->rows; ++r)
            f<<kernel->At(r,c)<<'\t';
        f<<endl;
    }
    f.close();
}


Ensemble::~Ensemble(){
    delete separateInput;
    delete separateInputDelta;
    delete separateOutput;
    delete separateOutputDelta;
    delete pooledInput;

    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(separateInput_r);
    DeleteOnlyShell(separateInputDelta_r);
    DeleteOnlyShell(separateOutput_r);
    DeleteOnlyShell(separateOutputDelta_r);
    DeleteOnlyShell(separateInput_rc);
    DeleteOnlyShell(separateInputDelta_rc);
    DeleteOnlyShell(separateOutput_rc);
    DeleteOnlyShell(separateOutputDelta_rc);
}






SymmetricEnsemble::SymmetricEnsemble(int weightsNum_, int lastLayers_): weightsNum(weightsNum_), lastLayers(lastLayers_){
}

void SymmetricEnsemble::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input = static_cast<tensor*>(layersData->layerList[from]);
    output = layersData->layerList[to];

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);

    kernel=    static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(   gradient->weightList[weightsNum].dataWeight);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialInput = new tensor();
    partialInput->SubLastTensor(input, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubLastTensor(inputDelta, lastLayers);

    separateInput = new tensor(input->rows, input->cols, lastLayers);
    separateInputDelta = new tensor(input->rows, input->cols, lastLayers);

    separateOutput = new tensor(input->rows, input->cols / 2, output->len);
    separateOutputDelta = new tensor(input->rows, input->cols / 2, output->len);

    pooledInput = new tensor(lastLayers, 1, 1);

    separateInput_r = new matrix();
    separateOutput_r = new matrix();
    separateInputDelta_r = new matrix();
    separateOutputDelta_r = new matrix();
    separateInput_rc = new vect();
    separateOutput_rc = new vect();
    separateInputDelta_rc = new vect();
    separateOutputDelta_rc = new vect();
    separateInput_r_1_c = new vect();
    separateInputDelta_r_1_c = new vect();

    if (lastLayers != kernel->cols ||
        output->len != kernel->rows ||
        lastLayers > input->depth)
        cout<<"Error in SymmetricEnsemble from layer "<<from<<" to layer "<<to<<endl;
}

void SymmetricEnsemble::ForwardPass(){
    if (testMode){
        AveragePool3D_all(partialInput, pooledInput);
        output->AddMatrVectProduct(kernel, pooledInput);
        SoftMax(output, output);
    }

    if (!testMode){
        separateInput->Rearrange(partialInput);
        separateOutput->SetToZero();
        for(int r=0; r<input->rows; ++r){
            separateInput_r->SetToTensorLayer(separateInput, r);
            separateOutput_r->SetToTensorLayer(separateOutput, r);
            for(int c = 0; c<input->cols / 2; ++c){
                separateInput_rc   ->SetToMatrixRow(separateInput_r, c);
                separateInput_r_1_c->SetToMatrixRow(separateInput_r, input->cols - 1 - c);
                separateInput_rc->AverageWith(separateInput_r_1_c);
                separateOutput_rc->SetToMatrixRow(separateOutput_r, c);
                separateOutput_rc->AddMatrVectProduct(kernel, separateInput_rc);
                SoftMax(separateOutput_rc, separateOutput_rc);
            }
        }
    }
}

void SymmetricEnsemble::BackwardPass(bool computeDelta, int trueClass){
    separateOutputDelta->Copy(separateOutput);
    for(int r = 0; r < input->rows; ++r)
        for(int c = 0; c < input->cols / 2; ++c)
            separateOutputDelta->At(r, c, trueClass) -= 1.0;
    separateOutputDelta->Multiply(2.0 / (input->rows * input->cols));

    separateInputDelta->SetToZero();
    for(int r=0; r<input->rows; ++r){
        separateInput_r->SetToTensorLayer(separateInput, r);
        separateInputDelta_r->SetToTensorLayer(separateInputDelta, r);
        separateOutputDelta_r->SetToTensorLayer(separateOutputDelta, r);
        for(int c = 0; c < input->cols / 2; ++c){
            separateInput_rc->SetToMatrixRow(separateInput_r, c);
            separateInputDelta_rc   ->SetToMatrixRow(separateInputDelta_r, c);
            separateInputDelta_r_1_c->SetToMatrixRow(separateInputDelta_r, input->cols - 1 - c);

            separateOutputDelta_rc->SetToMatrixRow(separateOutputDelta_r, c);
            separateInputDelta_rc->BackwardFullyConnectedNoBias(kernel, separateOutputDelta_rc, separateInput_rc, kernelGrad);
            separateInputDelta_rc->BackwardAverageWith(separateInputDelta_r_1_c);
        }
    }
    partialInputDelta->BackwardRearrange(separateInputDelta);

    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void SymmetricEnsemble::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SymmetricEnsemble::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode=1;
}

bool SymmetricEnsemble::HasWeightsDependency(){
    return 1;
}

void SymmetricEnsemble::WriteStructuredWeightsToFile(){
    char fileName[] = STRUCTERED_WEIGHTS;
    fileName[9] = '0' + weightsNum;
    ofstream f(fileName);
    f<<endl<<"kernel: "<<endl;
    for(int c=0; c<kernel->cols; ++c){
        for(int r=0; r<kernel->rows; ++r)
            f<<kernel->At(r,c)<<'\t';
        f<<endl;
    }
    f.close();
}


SymmetricEnsemble::~SymmetricEnsemble(){
    delete separateInput;
    delete separateInputDelta;
    delete separateOutput;
    delete separateOutputDelta;
    delete pooledInput;

    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(separateInput_r);
    DeleteOnlyShell(separateInputDelta_r);
    DeleteOnlyShell(separateOutput_r);
    DeleteOnlyShell(separateOutputDelta_r);
    DeleteOnlyShell(separateInput_rc);
    DeleteOnlyShell(separateInputDelta_rc);
    DeleteOnlyShell(separateOutput_rc);
    DeleteOnlyShell(separateOutputDelta_rc);
    DeleteOnlyShell(separateInput_r_1_c);
    DeleteOnlyShell(separateInputDelta_r_1_c);
}









Convolution2D2D::Convolution2D2D(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void Convolution2D2D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new matrix(kernel->rows, kernel->cols);

    bias=static_cast<realNumber*>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<realNumber *>(gradient->weightList[weightsNum].bias);

    indexInputRow.reserve(32);
    indexOutputRow.reserve(32);

    input=static_cast<matrix*>(layersData->layerList[from]);
    output=static_cast<matrix*>(layersData->layerList[to]);

    inputDelta=static_cast<matrix*>(deltas->layerList[from]);
    outputDelta=static_cast<matrix*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols)
        cout<<"Error in Convolution 2D2D from "<<from<<" to "<<to<<endl;
}



void Convolution2D2D::ForwardPass(){
    reversedKernel->Reverse(kernel);
    Convolute2D2D(input, output, reversedKernel, bias, paddingR, paddingC, indexInputRow);
    output->SetDroppedElementsToZero(outputActivity);
}

void Convolution2D2D::BackwardPass(bool computeDelta, int trueClass){
    if (computeDelta){
        BackwardConvolute2D2D(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else
        BackwardConvoluteGrad2D2D(input, outputDelta, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
}


bool Convolution2D2D::HasWeightsDependency(){
    return 1;
}


Convolution2D2D::~Convolution2D2D(){
    delete reversedKernel;
}




Convolution2D3D::Convolution2D3D(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void Convolution2D3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor(kernel->depth, kernel->rows, kernel->cols);

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    indexInputRow.reserve(32);
    indexOutputRow.reserve(32);

    input=static_cast<matrix*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<matrix*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols ||
        kernel->depth != output->depth ||
        bias->len != output->depth)
        cout<<"Error in Convolution 2D3D from "<<from<<" to "<<to<<endl;
}

void Convolution2D3D::ForwardPass(){
    reversedKernel->Reverse(kernel);
    Convolute2D3D(input, output, reversedKernel, bias, paddingR, paddingC, indexInputRow);
    output->SetDroppedElementsToZero(outputActivity);
}

void Convolution2D3D::BackwardPass(bool computeDelta, int trueClass){
    if (computeDelta){
        BackwardConvolute2D3D(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else
        BackwardConvoluteGrad2D3D(input, outputDelta, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
}

bool Convolution2D3D::HasWeightsDependency(){
    return 1;
}


Convolution2D3D::~Convolution2D3D(){
    delete reversedKernel;
}




Convolution3D2D::Convolution3D2D(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void Convolution3D2D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor(kernel->depth, kernel->rows, kernel->cols);

    bias=static_cast<realNumber *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<realNumber *>(gradient->weightList[weightsNum].bias);

    indexInputRow.reserve(32);
    indexOutputRow.reserve(32);

    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<matrix*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<matrix*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols ||
        kernel->depth != input->depth)
        cout<<"Error in Convolution 3D2D from "<<from<<" to "<<to<<endl;
}


void Convolution3D2D::ForwardPass(){
    reversedKernel->Reverse(kernel);
    Convolute3D2D(input, output, reversedKernel, bias, paddingR, paddingC, indexInputRow);
    output->SetDroppedElementsToZero(outputActivity);
}

void Convolution3D2D::BackwardPass(bool computeDelta, int trueClass){
    if (computeDelta){
        BackwardConvolute3D2D(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }
    else
        BackwardConvoluteGrad3D2D(input, outputDelta, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
}

bool Convolution3D2D::HasWeightsDependency(){
    return 1;
}


Convolution3D2D::~Convolution3D2D(){
    delete reversedKernel;
}




Convolution3D3D::Convolution3D3D(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void Convolution3D3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor4D *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor4D *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor4D(kernel->number, kernel->depth, kernel->rows, kernel->cols);

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    indexInputRow.reserve(32);
    indexOutputRow.reserve(32);

    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols ||
        kernel->depth != input->depth ||
        kernel->number != output->depth ||
        bias->len != output->depth)
        cout<<"Error in Convolution 3D3D from "<<from<<" to "<<to<<endl;
}

void Convolution3D3D::ForwardPass(){
    reversedKernel->Reverse(kernel);
    Convolute3D3D(input, output, reversedKernel, bias, paddingR, paddingC, indexInputRow);
    output->SetDroppedElementsToZero(outputActivity);
}

void Convolution3D3D::BackwardPass(bool computeDelta, int trueClass){
    if (computeDelta){
        BackwardConvolute3D3D(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else
        BackwardConvoluteGrad3D3D(input, outputDelta, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
}

bool Convolution3D3D::HasWeightsDependency(){
    return 1;
}


Convolution3D3D::~Convolution3D3D(){
    delete reversedKernel;
}



PointwiseFunctionLayer::PointwiseFunctionLayer(mathFunc* func_){
    func=func_;
}

void PointwiseFunctionLayer::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input  = layersData->layerList[from];
    output = layersData->layerList[to];

    inputDelta = deltas->layerList[from];
    outputDelta= deltas->layerList[to];

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->len != output->len)
        cout<<"Error in pointwise function layer from "<<from<<" to "<<to<<endl;
}

void PointwiseFunctionLayer::ForwardPass(){
    output->AddPointwiseFunction(input, func);
    output->SetDroppedElementsToZero(outputActivity);
}

void PointwiseFunctionLayer::BackwardPass(bool computeDelta, int trueClass){
    if (computeDelta){
        inputDelta->AddPointwiseFuncDerivMultiply(outputDelta, input, func);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }
}

bool PointwiseFunctionLayer::HasWeightsDependency(){
    return 0;
}


SoftMaxLayer::SoftMaxLayer(){
}

void SoftMaxLayer::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<vect*>(layersData->layerList[from]);
    output=static_cast<vect*>(layersData->layerList[to]);

    if (input->len != output->len)
        cout<<"Error in SoftMaxLayer from "<<from<<" to "<<to<<endl;
}

void SoftMaxLayer::ForwardPass(){
    SoftMax(input, output);
}

void SoftMaxLayer::BackwardPass(bool computeDelta, int trueClass){
    cout<<"Backward for softmax: something is wrong"<<endl;
}

bool SoftMaxLayer::HasWeightsDependency(){
    return 0;
}




AveragePooling2D::AveragePooling2D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void AveragePooling2D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input =static_cast<matrix*>(layersData->layerList[from]);
    output=static_cast<matrix*>(layersData->layerList[to]);

    inputDelta  = static_cast<matrix*>(deltas->layerList[from]);
    outputDelta = static_cast<matrix*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (output->rows * kernelRsize != input->rows || output->cols * kernelCsize != input->cols)
        cout<<"Error in average pooling 2D from "<<from<<" to "<<to<<endl;
}

void AveragePooling2D::ForwardPass(){
    AveragePool2D(input, output, kernelRsize, kernelCsize);
    output->SetDroppedElementsToZero(outputActivity);
}

void AveragePooling2D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;

    BackwardAveragePool2D(inputDelta, outputDelta, kernelRsize, kernelCsize);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}


bool AveragePooling2D::HasWeightsDependency(){
    return 0;
}




AveragePooling3D::AveragePooling3D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void AveragePooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input =static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta  = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols ||
        output->depth != input->depth)
        cout<<"Error in average pooling 3D from "<<from<<" to "<<to<<endl;
}

void AveragePooling3D::ForwardPass(){
    AveragePool3D(input, output, kernelRsize, kernelCsize);
    output->SetDroppedElementsToZero(outputActivity);
}

void AveragePooling3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    BackwardAveragePool3D(inputDelta, outputDelta, kernelRsize, kernelCsize);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}


bool AveragePooling3D::HasWeightsDependency(){
    return 0;
}






Merge::Merge(int startingIndex_): startingIndex(startingIndex_){
}

void Merge::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input = layersData->layerList[from];
    output= layersData->layerList[to];

    inputDelta  = deltas->layerList[from];
    outputDelta = deltas->layerList[to];

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (startingIndex + input->len > output->len)
        cout<<"Error in Merge from "<<from<<" to "<<to<<endl;
}

void Merge::ForwardPass(){
    output->AddThisStartingFromOnlyActive(startingIndex, input, outputActivity);
}

void Merge::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    inputDelta->AddAddonStartingFromOnlyActive(startingIndex, outputDelta, inputActivity);
}

bool Merge::HasWeightsDependency(){
    return 0;
}



ConvoluteReluMerge::ConvoluteReluMerge(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void ConvoluteReluMerge::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor4D *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor4D *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor4D(kernel->number, kernel->depth, kernel->rows, kernel->cols);

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    indexInputRow.reserve(32);
    indexOutputRow.reserve(32);

    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    convolutionOutput = new tensor();
    convolutionOutputDelta = new tensor();

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols ||
        kernel->depth != input->depth ||
        kernel->number + input->depth != output->depth ||
        bias->len != kernel->number)
        cout<<"Error in ConvoluteReluMerge from "<<from<<" to "<<to<<endl;
}

void ConvoluteReluMerge::ForwardPass(){
    convolutionOutput -> SubTensor(output, kernel->number);
    reversedKernel->Reverse(kernel);
    Convolute3D3D(input, convolutionOutput, reversedKernel, bias, paddingR, paddingC, indexInputRow);

    convolutionOutput->SetToReluFunction();

    int startingIndex = convolutionOutput->len;
    output->AddThisStartingFrom(startingIndex, input);
    output->SetDroppedElementsToZero(outputActivity);
}

void ConvoluteReluMerge::BackwardPass(bool computeDelta, int trueClass){
    convolutionOutputDelta -> SubTensor(outputDelta, kernel->number);
    int startingIndex = convolutionOutputDelta->len;

    for(int j=0; j<convolutionOutputDelta->len; ++j)
        convolutionOutputDelta->elem[j] *= (output->elem[j]>0);

    if (computeDelta){
        BackwardConvolute3D3D(input, inputDelta, convolutionOutputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->AddAddonStartingFrom(startingIndex, outputDelta);
    }
    else
        BackwardConvoluteGrad3D3D(input, convolutionOutputDelta, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);

    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool ConvoluteReluMerge::HasWeightsDependency(){
    return 1;
}


ConvoluteReluMerge::~ConvoluteReluMerge(){
    delete reversedKernel;
    DeleteOnlyShell(convolutionOutput);
    DeleteOnlyShell(convolutionOutputDelta);
}







ConvoluteMaxMinMerge::ConvoluteMaxMinMerge(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void ConvoluteMaxMinMerge::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor4D *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor4D *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor4D(kernel->number, kernel->depth, kernel->rows, kernel->cols);

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    convolutionOutput = new tensor();
    convolutionOutput -> SubTensor(output, kernel->number);

    minOutput = new tensor();
    minOutput->SubTensor(output, kernel->number, kernel->number);

    convolutionOutputDelta = new tensor();
    convolutionOutputDelta -> SubTensor(outputDelta, kernel->number);

    minOutputDelta = new tensor();
    minOutputDelta->SubTensor(outputDelta, kernel->number, kernel->number);

    indexInputRow.reserve(input->cols);
    indexOutputRow.reserve(output->cols);

    startingIndexMerge = 2 * convolutionOutput->len;

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols ||
        kernel->depth != input->depth ||
        2*kernel->number + input->depth != output->depth ||
        bias->len != kernel->number)
        cout<<"Error in ConvoluteMaxMin from "<<from<<" to "<<to<<endl;
}



void ConvoluteMaxMinMerge::ForwardPass(){
    reversedKernel->Reverse(kernel);
    Convolute3D3D(input, convolutionOutput, reversedKernel, bias, paddingR, paddingC, indexInputRow);

    minOutput->SetToMinReluFunction(convolutionOutput);
    convolutionOutput->SetToReluFunction();
    output->AddThisStartingFrom(startingIndexMerge, input);

    output->SetDroppedElementsToZero(outputActivity);
}

void ConvoluteMaxMinMerge::BackwardPass(bool computeDelta, int trueClass){
    convolutionOutputDelta->MaxMinBackward(minOutputDelta, output);

    if (computeDelta){
        BackwardConvolute3D3D(input, inputDelta, convolutionOutputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->AddAddonStartingFrom(startingIndexMerge, outputDelta);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else
        BackwardConvoluteGrad3D3D(input, convolutionOutputDelta, kernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
}

bool ConvoluteMaxMinMerge::HasWeightsDependency(){
    return 1;
}


ConvoluteMaxMinMerge::~ConvoluteMaxMinMerge(){
    delete reversedKernel;
    DeleteOnlyShell(convolutionOutput);
    DeleteOnlyShell(minOutput);
    DeleteOnlyShell(convolutionOutputDelta);
    DeleteOnlyShell(minOutputDelta);
}





SequentiallyConvoluteMaxMin::SequentiallyConvoluteMaxMin(int weightsNum_, int startDepth_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), startDepth(startDepth_), paddingR(paddingR_), paddingC(paddingC_){
}

void SequentiallyConvoluteMaxMin::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor(kernel->depth, kernel->rows, kernel->cols);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    int nConvolutions = bias->len;

    indexInputRow.reserve(input->cols);

    if (from != to ||
        input->rows + 2*paddingR - kernel->rows + 1 != input->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != input->cols ||
        nConvolutions * (startDepth + nConvolutions - 1) != kernel->depth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteMaxMin from "<<from<<" to "<<to<<endl;
}

void SequentiallyConvoluteMaxMin::ForwardPass(){
    reversedKernel->Reverse(kernel);
    SequentialConvolution(input, reversedKernel, bias, paddingR, paddingC, indexInputRow, startDepth, inputActivity);
}

void SequentiallyConvoluteMaxMin::BackwardPass(bool computeDelta, int trueClass){
   BackwardSequentialConvolution(input, inputDelta, kernel, kernelGrad, biasGrad, paddingR, paddingC, indexInputRow, startDepth, computeDelta, inputActivity);
}

bool SequentiallyConvoluteMaxMin::HasWeightsDependency(){
    return 1;
}


SequentiallyConvoluteMaxMin::~SequentiallyConvoluteMaxMin(){
    delete reversedKernel;
}




SequentiallyConvoluteMaxMinStandard::SequentiallyConvoluteMaxMinStandard(int weightsNum_, int startDepth_):
    weightsNum(weightsNum_), startDepth(startDepth_){
}

void SequentiallyConvoluteMaxMinStandard::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    int nConvolutions = bias->len;

    if (from != to ||
        kernel->rows!=3||
        kernel->cols!=3||
        nConvolutions * (startDepth + nConvolutions - 1) != kernel->depth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteMaxMin from "<<from<<" to "<<to<<endl;
}

void SequentiallyConvoluteMaxMinStandard::ForwardPass(){
    SequentialConvolutionStandard(input, kernel, bias, startDepth, inputActivity);
}

void SequentiallyConvoluteMaxMinStandard::BackwardPass(bool computeDelta, int trueClass){
   BackwardSequentialConvolutionStandard(input, inputDelta, kernel, kernelGrad, biasGrad, startDepth, computeDelta, inputActivity);
}

bool SequentiallyConvoluteMaxMinStandard::HasWeightsDependency(){
    return 1;
}


SequentiallyConvoluteMaxMinStandard::~SequentiallyConvoluteMaxMinStandard(){
}









SequentiallyConvoluteBottleneckReluStandard::SequentiallyConvoluteBottleneckReluStandard(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), nConvolutions(nConvolutions_){
}

void SequentiallyConvoluteBottleneckReluStandard::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=3||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        nConvolutions * (2 * startDepth + nConvolutions - 1) / 2 != kernel_vert->depth ||
        input->depth != startDepth + nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckReluStandard from "<<from<<" to "<<to<<endl;
}

void SequentiallyConvoluteBottleneckReluStandard::ForwardPass(){
    SequentialBottleneckConvolutionReluStandard(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions, inputActivity);
}

void SequentiallyConvoluteBottleneckReluStandard::BackwardPass(bool computeDelta, int trueClass){
   BackwardSequentialBottleneckConvolutionReluStandard(input, inputDelta, kernel_vert, kernelGrad_vert,
                                                   kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                   verticalConvDelta, startDepth, nConvolutions, inputActivity);
}

void SequentiallyConvoluteBottleneckReluStandard::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
}

void SequentiallyConvoluteBottleneckReluStandard::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
}

bool SequentiallyConvoluteBottleneckReluStandard::HasWeightsDependency(){
    return 1;
}

SequentiallyConvoluteBottleneckReluStandard::~SequentiallyConvoluteBottleneckReluStandard(){
    delete verticalConv;
    delete verticalConvDelta;
}





SequentiallyConvoluteBottleneckMaxMinStandard::SequentiallyConvoluteBottleneckMaxMinStandard(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), nConvolutions(nConvolutions_){
}

void SequentiallyConvoluteBottleneckMaxMinStandard::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    firstLayer = (from == 0);

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=3||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        nConvolutions * (startDepth + nConvolutions - 1) != kernel_vert->depth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandard from "<<from<<" to "<<to<<endl;
}

void SequentiallyConvoluteBottleneckMaxMinStandard::ForwardPass(){
    SequentialBottleneckConvolutionStandard(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions, inputActivity, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandard::BackwardPass(bool computeDelta, int trueClass){
    //computeDelta = (from>0);
    if (!PARALLEL_ARCHITECTURE)
    {
        if (!firstLayer)
            BackwardSequentialBottleneckConvolutionStandard (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                            kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                            verticalConvDelta, startDepth, nConvolutions, inputActivity);
        else
            BackwardSequentialBottleneckConvolutionStandardPartialGrad  (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                                        kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                                        verticalConvDelta, startDepth, nConvolutions, inputActivity);
    }
    else
        BackwardSequentialBottleneckConvolutionStandardPartialGrad  (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                                        kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                                        verticalConvDelta, startDepth, nConvolutions, inputActivity);
}

void SequentiallyConvoluteBottleneckMaxMinStandard::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandard::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandard::HasWeightsDependency(){
    return 1;
}


SequentiallyConvoluteBottleneckMaxMinStandard::~SequentiallyConvoluteBottleneckMaxMinStandard(){
    delete verticalConv;
    delete verticalConvDelta;
}







SequentiallyMaxMinStandard::SequentiallyMaxMinStandard(int weightsNum_, int startDepth_, int nConvolutions_):
    weightsNum(weightsNum_), startDepth(startDepth_), nConvolutions(nConvolutions_){
}

void SequentiallyMaxMinStandard::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;

    kernel = static_cast<vect *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad = static_cast<vect *>(gradient->weightList[weightsNum].dataWeight);

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    if (from != to ||
        bias->len != nConvolutions||
        kernel->len != nConvolutions * (startDepth + nConvolutions - 1)||
        input->rows != 1 ||
        input->cols != 1 ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyMaxMinStandard from "<<from<<" to "<<to<<endl;
}

void SequentiallyMaxMinStandard::ForwardPass(){
    SequentialMaxMinStandard(input, kernel, bias, startDepth, nConvolutions, inputActivity, testMode);
}

void SequentiallyMaxMinStandard::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialMaxMinStandard(input, inputDelta, kernel, kernelGrad, biasGrad, startDepth, nConvolutions, inputActivity);
}

void SequentiallyMaxMinStandard::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyMaxMinStandard::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyMaxMinStandard::HasWeightsDependency(){
    return 1;
}


SequentiallyMaxMinStandard::~SequentiallyMaxMinStandard(){
}









SequentiallyConvoluteBottleneckMaxMinStandardLimited::SequentiallyConvoluteBottleneckMaxMinStandardLimited
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_),
    startDepth(startDepth_), nConvolutions(nConvolutions_), limitDepth(limitDepth_), alwaysPresentDepth(alwaysPresentDepth_){
}

void SequentiallyConvoluteBottleneckMaxMinStandardLimited::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
    //cout<<"Check vertLen: "<<vertLen<<" nCOnv: "<<nConvolutions<<endl;
    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=3||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        startDepth < alwaysPresentDepth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandardLimited from "<<from<<" to "<<to<<endl;
}

void SequentiallyConvoluteBottleneckMaxMinStandardLimited::ForwardPass(){
    SequentialBottleneckConvolutionStandardLimited(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions, limitDepth, alwaysPresentDepth, inputActivity, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandardLimited::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialBottleneckConvolutionStandardLimited (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                            kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                            verticalConvDelta, startDepth, nConvolutions, limitDepth, alwaysPresentDepth, inputActivity);
}

void SequentiallyConvoluteBottleneckMaxMinStandardLimited::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandardLimited::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardLimited::HasWeightsDependency(){
    return 1;
}


SequentiallyConvoluteBottleneckMaxMinStandardLimited::~SequentiallyConvoluteBottleneckMaxMinStandardLimited(){
    delete verticalConv;
    delete verticalConvDelta;
}







SequentiallyConvoluteBottleneckMaxMinStandardRandom::SequentiallyConvoluteBottleneckMaxMinStandardRandom
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_),
    startDepth(startDepth_), nConvolutions(nConvolutions_), limitDepth(limitDepth_), alwaysPresentDepth(alwaysPresentDepth_){
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandom::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);

    indices = new int[vertLen];

    int * indices_j = indices;
    int trueAdditionalLen;
    int * pairIndices = new int[limitDepth / 2];

    for(int j=0; j<nConvolutions; ++j){
        for(int k=0; k<alwaysPresentDepth; ++k)
            indices_j[k] = k;
        indices_j += alwaysPresentDepth;
        trueAdditionalLen = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        FillRandom(pairIndices, (alwaysPresentDepth + 1) / 2, (startDepth + 2 * j + 1) / 2, trueAdditionalLen / 2);
        for(int k=0; k<trueAdditionalLen / 2; ++k){
            indices_j[2 * k]        = 2 * pairIndices[k] - 1;
            indices_j[2 * k + 1]    = 2 * pairIndices[k];
        }
        indices_j += trueAdditionalLen;
    }

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=3||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        startDepth < alwaysPresentDepth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandardRandom from "<<from<<" to "<<to<<endl;

    delete [] pairIndices;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandom::ForwardPass(){
    SequentialBottleneckConvolutionStandardRandom(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions,
                                                   limitDepth, alwaysPresentDepth, inputActivity, indices, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandom::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialBottleneckConvolutionStandardRandom (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                            kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                            verticalConvDelta, startDepth, nConvolutions, limitDepth, alwaysPresentDepth, inputActivity, indices);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandom::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandom::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandom::HasWeightsDependency(){
    return 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandom::NeedsUnification(){
    return 1;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandom::Unify(computationalNode * primalCN){
    SequentiallyConvoluteBottleneckMaxMinStandardRandom * primalNode = static_cast<SequentiallyConvoluteBottleneckMaxMinStandardRandom *> (primalCN);
    for(int j=0; j<kernel_vert->depth; ++j)
        indices[j] = primalNode->indices[j];
}

SequentiallyConvoluteBottleneckMaxMinStandardRandom::~SequentiallyConvoluteBottleneckMaxMinStandardRandom(){
    delete verticalConv;
    delete verticalConvDelta;
    delete []indices;
}









SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_, bool innerDropping_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), nConvolutions(nConvolutions_),
    limitDepth(limitDepth_), alwaysPresentDepth(alwaysPresentDepth_), innerDropping(innerDropping_){
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvActivity = new activityData(verticalConv->len, inputActivity->dropRate * innerDropping);

    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);

    indices = new int[vertLen];

    int * indices_j = indices;
    int trueAdditionalLen;
    int * pairIndices = new int[limitDepth / 2];

    for(int j=0; j<nConvolutions; ++j){
        for(int k=0; k<alwaysPresentDepth; ++k)
            indices_j[k] = k;
        indices_j += alwaysPresentDepth;
        trueAdditionalLen = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        FillRandom(pairIndices, (alwaysPresentDepth + 1) / 2, (startDepth + 2 * j + 1) / 2, trueAdditionalLen / 2);
        for(int k=0; k<trueAdditionalLen / 2; ++k){
            indices_j[2 * k]        = 2 * pairIndices[k] - 1;
            indices_j[2 * k + 1]    = 2 * pairIndices[k];
        }
        indices_j += trueAdditionalLen;
    }

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=2||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        startDepth < alwaysPresentDepth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric from "<<from<<" to "<<to<<endl;

    delete [] pairIndices;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::ForwardPass(){
    if (!testMode)
        verticalConvActivity->DropUnits();
    SequentialBottleneckConvolutionStandardRandomSymmetric(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions,
                                                   limitDepth, alwaysPresentDepth, inputActivity, verticalConvActivity, indices, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialBottleneckConvolutionStandardRandomSymmetric (input, inputDelta, kernel_vert, kernelGrad_vert,
                                            kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv, verticalConvDelta, startDepth,
                                            nConvolutions, limitDepth, alwaysPresentDepth, inputActivity, verticalConvActivity, indices);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
       kernel_hor ->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
       kernel_hor ->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::HasWeightsDependency(){
    return 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::NeedsUnification(){
    return 1;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::Unify(computationalNode * primalCN){
    SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric * primalNode = static_cast<SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric *> (primalCN);
    for(int j=0; j<kernel_vert->depth; ++j)
        indices[j] = primalNode->indices[j];
}

SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric::~SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric(){
    delete verticalConv;
    delete verticalConvDelta;
    delete verticalConvActivity;
    delete []indices;
}







SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_, bool innerDropping_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), nConvolutions(nConvolutions_),
    limitDepth(limitDepth_), alwaysPresentDepth(alwaysPresentDepth_), innerDropping(innerDropping_){
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvActivity = new activityData(verticalConv->len, inputActivity->dropRate * innerDropping);

    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);

    indices = new int[vertLen];

    int * indices_j = indices;
    int trueAdditionalLen;
    int * pairIndices = new int[limitDepth / 2];

    for(int j=0; j<nConvolutions; ++j){
        for(int k=0; k<alwaysPresentDepth; ++k)
            indices_j[k] = k;
        indices_j += alwaysPresentDepth;
        trueAdditionalLen = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        FillRandom(pairIndices, (alwaysPresentDepth + 1) / 2, (startDepth + 2 * j + 1) / 2, trueAdditionalLen / 2);
        for(int k=0; k<trueAdditionalLen / 2; ++k){
            indices_j[2 * k]        = 2 * pairIndices[k] - 1;
            indices_j[2 * k + 1]    = 2 * pairIndices[k];
        }
        indices_j += trueAdditionalLen;
    }

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=2||
        kernel_hor->depth!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        startDepth < alwaysPresentDepth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias from "<<from<<" to "<<to<<endl;

    delete [] pairIndices;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::ForwardPass(){
    if (!testMode)
        verticalConvActivity->DropUnits();
    SequentialBottleneckConvolutionStandardRandomSymmetricNoBias(input, kernel_vert, kernel_hor, verticalConv, startDepth, nConvolutions,
                                                   limitDepth, alwaysPresentDepth, inputActivity, verticalConvActivity, indices, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialBottleneckConvolutionStandardRandomSymmetricNoBias (input, inputDelta, kernel_vert, kernelGrad_vert,
                                            kernel_hor, kernelGrad_hor, verticalConv, verticalConvDelta, startDepth,
                                            nConvolutions, limitDepth, alwaysPresentDepth, inputActivity, verticalConvActivity, indices);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
       kernel_hor ->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
       kernel_hor ->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::HasWeightsDependency(){
    return 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::NeedsUnification(){
    return 1;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::Unify(computationalNode * primalCN){
    SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias * primalNode = static_cast<SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias *> (primalCN);
    for(int j=0; j<kernel_vert->depth; ++j)
        indices[j] = primalNode->indices[j];
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::WriteStructuredWeightsToFile(){
    char fileName[] = STRUCTERED_WEIGHTS;
    fileName[9] = '0' + weightsNum_horizontal;
    ofstream f(fileName);
    matrix* kernel_hor_j = new matrix();
    for(int j=0; j<nConvolutions; ++j){
        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
        f<<"Kernel "<<j<<":"<<endl;
        for(int r=0; r<3; ++r){
            f<<kernel_hor_j->At(r, 0)<<'\t'<<kernel_hor_j->At(r, 1)<<'\t'<<kernel_hor_j->At(r, 0)<<endl;
        }
        f<<endl;
    }
    f.close();
    DeleteOnlyShell(kernel_hor_j);

    fileName[9] = '0' + weightsNum_vertical;
    f.open(fileName);

    tensor* kernel_vert_j = new tensor();
    int vertKernelStartIndex = 0;
    for(int j=0; j<nConvolutions; ++j){
        kernel_vert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        f<<"Vertical coefficients "<<j<<":"<<endl;
        for(int c=0; c<kernel_vert_j->len; ++c)
            f<<kernel_vert_j->elem[c]<<endl;
        f<<endl;
    }
    f.close();
    DeleteOnlyShell(kernel_vert_j);


}

SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias::~SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias(){
    delete verticalConv;
    delete verticalConvDelta;
    delete verticalConvActivity;
    delete []indices;
}







StairsConvolution::StairsConvolution
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_, bool innerDropping_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), numStairs(numStairs_),
    numStairConvolutions(numStairConvolutions_), innerDropping(innerDropping_){
}

void StairsConvolution::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + 2 * numStairConvolutions * stair) * numStairConvolutions;
    }

    int nConvolutions = numStairs * numStairConvolutions;

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvActivity = new activityData(verticalConv->len, inputActivity->dropRate * innerDropping);

    indices = new int [vertLen];
    int * indices_conv = indices;
    for(int stair=0; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            for(int j=0; j<startDepth + 2 * numStairConvolutions * stair; ++j)
                indices_conv[j] = j;
            indices_conv += startDepth + 2 * numStairConvolutions * stair;
        }

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=2||
        kernel_hor->depth!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in StairsConvolution from "<<from<<" to "<<to<<endl;
}

void StairsConvolution::ForwardPass(){
    if (!testMode)
        verticalConvActivity->DropUnits();
    ForwardStairsConvolution(input, kernel_vert, kernel_hor, verticalConv, startDepth, numStairs, numStairConvolutions,
                             inputActivity, verticalConvActivity, indices, testMode);
}

void StairsConvolution::BackwardPass(bool computeDelta, int trueClass){
    BackwardStairsConvolution(input, inputDelta, kernel_vert, kernelGrad_vert, kernel_hor, kernelGrad_hor, verticalConv, verticalConvDelta,
                            startDepth, numStairs, numStairConvolutions, inputActivity, verticalConvActivity, indices);
}

void StairsConvolution::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Stairs Convolution is already in train mode"<<endl;
        return;
    }

    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
       if (verticalConvActivity->dropping)
            kernel_hor ->Multiply(1.0/(1.0-verticalConvActivity->dropRate));
    }
    testMode = 0;
}

void StairsConvolution::SetToTestMode(){
    if (testMode==1){
        cout<<"Stairs Convolution is already in test mode"<<endl;
        return;
    }
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
       if (verticalConvActivity->dropping)
            kernel_hor ->Multiply(1.0-verticalConvActivity->dropRate);
    }
    testMode = 1;
}

bool StairsConvolution::HasWeightsDependency(){
    return 1;
}

bool StairsConvolution::NeedsUnification(){
    return 0;
}

void StairsConvolution::Unify(computationalNode * primalCN){
}

void StairsConvolution::WriteStructuredWeightsToFile(){
//    char fileName[] = STRUCTERED_WEIGHTS;
//    fileName[9] = '0' + weightsNum_horizontal;
//    ofstream f(fileName);
//    matrix* kernel_hor_j = new matrix();
//    for(int j=0; j<nConvolutions; ++j){
//        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
//        f<<"Kernel "<<j<<":"<<endl;
//        for(int r=0; r<3; ++r){
//            f<<kernel_hor_j->At(r, 0)<<'\t'<<kernel_hor_j->At(r, 1)<<'\t'<<kernel_hor_j->At(r, 0)<<endl;
//        }
//        f<<endl;
//    }
//    f.close();
//    DeleteOnlyShell(kernel_hor_j);
//
//    fileName[9] = '0' + weightsNum_vertical;
//    f.open(fileName);
//
//    tensor* kernel_vert_j = new tensor();
//    int vertKernelStartIndex = 0;
//    for(int j=0; j<nConvolutions; ++j){
//        kernel_vert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
//        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
//        f<<"Vertical coefficients "<<j<<":"<<endl;
//        for(int c=0; c<kernel_vert_j->len; ++c)
//            f<<kernel_vert_j->elem[c]<<endl;
//        f<<endl;
//    }
//    f.close();
//    DeleteOnlyShell(kernel_vert_j);


}

StairsConvolution::~StairsConvolution(){
    delete verticalConv;
    delete verticalConvDelta;
    delete verticalConvActivity;
    delete []indices;
}






StairsSymmetricConvolution::StairsSymmetricConvolution
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), numStairs(numStairs_),
    numStairConvolutions(numStairConvolutions_), symmetryLevel(symmetryLevel_){
}

void StairsSymmetricConvolution::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + 2 * numStairConvolutions * stair) * numStairConvolutions;
    }

    int nConvolutions = numStairs * numStairConvolutions;

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    indices = new int [vertLen];
    int * indices_conv = indices;
    for(int stair=0; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            for(int j=0; j<startDepth + 2 * numStairConvolutions * stair; ++j)
                indices_conv[j] = j;
            indices_conv += startDepth + 2 * numStairConvolutions * stair;
        }

    if (symmetryLevel<0 || symmetryLevel>4)
        cout<<"Symmetry level is not implemented yet";

    if (from != to ||
        kernel_hor->depth!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error0 in StairsSymmetricConvolution from "<<from<<" to "<<to<<endl;

    if ((symmetryLevel == 0 && (kernel_hor->rows != 3 || kernel_hor->cols != 3) ) ||
        (symmetryLevel == 1 && (kernel_hor->rows != 3 || kernel_hor->cols != 2) ) ||
        (symmetryLevel == 2 && (kernel_hor->rows != 2 || kernel_hor->cols != 2) ) ||
        (symmetryLevel == 3 && (kernel_hor->rows != 1 || kernel_hor->cols != 3) ) ||
        (symmetryLevel == 4 && (kernel_hor->rows != 1 || kernel_hor->cols != 2) ))
        cout<<"Error1 in StairsSymmetricConvolution from "<<from<<" to "<<to<<endl;

}

void StairsSymmetricConvolution::ForwardPass(){
    ForwardStairsSymmetricConvolution(input, kernel_vert, kernel_hor, verticalConv, startDepth, numStairs, numStairConvolutions,
                             inputActivity, indices, testMode, symmetryLevel);
}

void StairsSymmetricConvolution::BackwardPass(bool computeDelta, int trueClass){
    BackwardStairsSymmetricConvolution(input, inputDelta, kernel_vert, kernelGrad_vert, kernel_hor, kernelGrad_hor, verticalConv, verticalConvDelta,
                            startDepth, numStairs, numStairConvolutions, inputActivity, indices, symmetryLevel);
}

void StairsSymmetricConvolution::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Stairs Convolution is already in train mode"<<endl;
        return;
    }

    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void StairsSymmetricConvolution::SetToTestMode(){
    if (testMode==1){
        cout<<"Stairs Convolution is already in test mode"<<endl;
        return;
    }
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool StairsSymmetricConvolution::HasWeightsDependency(){
    return 1;
}

bool StairsSymmetricConvolution::NeedsUnification(){
    return 0;
}

void StairsSymmetricConvolution::Unify(computationalNode * primalCN){
}

void StairsSymmetricConvolution::WriteStructuredWeightsToFile(){
//    char fileName[] = STRUCTERED_WEIGHTS;
//    fileName[9] = '0' + weightsNum_horizontal;
//    ofstream f(fileName);
//    matrix* kernel_hor_j = new matrix();
//    for(int j=0; j<nConvolutions; ++j){
//        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
//        f<<"Kernel "<<j<<":"<<endl;
//        for(int r=0; r<3; ++r){
//            f<<kernel_hor_j->At(r, 0)<<'\t'<<kernel_hor_j->At(r, 1)<<'\t'<<kernel_hor_j->At(r, 0)<<endl;
//        }
//        f<<endl;
//    }
//    f.close();
//    DeleteOnlyShell(kernel_hor_j);
//
//    fileName[9] = '0' + weightsNum_vertical;
//    f.open(fileName);
//
//    tensor* kernel_vert_j = new tensor();
//    int vertKernelStartIndex = 0;
//    for(int j=0; j<nConvolutions; ++j){
//        kernel_vert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
//        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
//        f<<"Vertical coefficients "<<j<<":"<<endl;
//        for(int c=0; c<kernel_vert_j->len; ++c)
//            f<<kernel_vert_j->elem[c]<<endl;
//        f<<endl;
//    }
//    f.close();
//    DeleteOnlyShell(kernel_vert_j);


}

StairsSymmetricConvolution::~StairsSymmetricConvolution(){
    delete verticalConv;
    delete verticalConvDelta;
    delete []indices;
}








StairsSymmetricConvolutionRelu::StairsSymmetricConvolutionRelu
(int weightsNum_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_):
    weightsNum(weightsNum_), startDepth(startDepth_), numStairs(numStairs_),
    numStairConvolutions(numStairConvolutions_), symmetryLevel(symmetryLevel_){
}

void StairsSymmetricConvolutionRelu::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;

    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + numStairConvolutions * stair) * numStairConvolutions;
    }

    if (symmetryLevel<0 || symmetryLevel>4)
        cout<<"Symmetry level is not implemented yet";

    if (from != to ||
        kernel->depth!=vertLen||
        input->depth != startDepth + numStairs * numStairConvolutions )
        cout<<"Error0 in StairsSymmetricConvolutionRelu from "<<from<<" to "<<to<<endl;

    if ((symmetryLevel == 0 && (kernel->rows != 3 || kernel->cols != 3) ) ||
        (symmetryLevel == 1 && (kernel->rows != 3 || kernel->cols != 2) ) ||
        (symmetryLevel == 2 && (kernel->rows != 2 || kernel->cols != 2) ) ||
        (symmetryLevel == 3 && (kernel->rows != 1 || kernel->cols != 3) ) ||
        (symmetryLevel == 4 && (kernel->rows != 1 || kernel->cols != 2) ))
        cout<<"Error1 in StairsSymmetricConvolutionRelu from "<<from<<" to "<<to<<endl;

}

void StairsSymmetricConvolutionRelu::ForwardPass(){
    ForwardStairsSymmetricConvolutionRelu(input, kernel, startDepth, numStairs, numStairConvolutions,
                             inputActivity, testMode, symmetryLevel);
}

void StairsSymmetricConvolutionRelu::BackwardPass(bool computeDelta, int trueClass){
    BackwardStairsSymmetricConvolutionRelu(input, inputDelta, kernel, kernelGrad,
                startDepth, numStairs, numStairConvolutions, inputActivity, symmetryLevel);
}

void StairsSymmetricConvolutionRelu::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Stairs Convolution is already in train mode"<<endl;
        return;
    }

    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void StairsSymmetricConvolutionRelu::SetToTestMode(){
    if (testMode==1){
        cout<<"Stairs Convolution is already in test mode"<<endl;
        return;
    }
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool StairsSymmetricConvolutionRelu::HasWeightsDependency(){
    return 1;
}

bool StairsSymmetricConvolutionRelu::NeedsUnification(){
    return 0;
}

void StairsSymmetricConvolutionRelu::Unify(computationalNode * primalCN){
}

void StairsSymmetricConvolutionRelu::WriteStructuredWeightsToFile(){
}

StairsSymmetricConvolutionRelu::~StairsSymmetricConvolutionRelu(){
}








StairsFullConvolution::StairsFullConvolution
(int weightsNum_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_, bool biasIncluded_):
    weightsNum(weightsNum_), startDepth(startDepth_), numStairs(numStairs_),
    numStairConvolutions(numStairConvolutions_), symmetryLevel(symmetryLevel_), biasIncluded(biasIncluded_){
}

void StairsFullConvolution::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;

    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);

    bias = static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad = static_cast<vect *>(gradient->weightList[weightsNum].bias);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + 2 * numStairConvolutions * stair) * numStairConvolutions;
    }

    if (symmetryLevel<0 || symmetryLevel>4)
        cout<<"Symmetry level is not implemented yet";

    if (from != to ||
        kernel->depth!=vertLen||
        input->depth != startDepth + 2 * numStairs * numStairConvolutions )
        cout<<"Error0 in StairsFullConvolution from "<<from<<" to "<<to<<endl;

    if ((symmetryLevel == 0 && (kernel->rows != 3 || kernel->cols != 3) ) ||
        (symmetryLevel == 1 && (kernel->rows != 3 || kernel->cols != 2) ) ||
        (symmetryLevel == 2 && (kernel->rows != 2 || kernel->cols != 2) ) ||
        (symmetryLevel == 3 && (kernel->rows != 1 || kernel->cols != 3) ) ||
        (symmetryLevel == 4 && (kernel->rows != 1 || kernel->cols != 2) ))
        cout<<"Error1 in StairsFullConvolution from "<<from<<" to "<<to<<endl;

    if (biasIncluded){
        if (bias->len != numStairs * numStairConvolutions)
            cout<<"Bias error in StairsFullConvolution from "<<from<<" to "<<to<<endl;
    }
    else{
        if (bias->len != 0)
            cout<<"Bias error in StairsFullConvolution from "<<from<<" to "<<to<<endl;
    }

}

void StairsFullConvolution::ForwardPass(){
    ForwardStairsFullConvolution(input, kernel, bias, startDepth, numStairs, numStairConvolutions,
                             inputActivity, testMode, symmetryLevel);
}

void StairsFullConvolution::BackwardPass(bool computeDelta, int trueClass){
    BackwardStairsFullConvolution(input, inputDelta, kernel, kernelGrad, biasGrad,
                startDepth, numStairs, numStairConvolutions, inputActivity, symmetryLevel);
}

void StairsFullConvolution::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Stairs Convolution is already in train mode"<<endl;
        return;
    }

    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void StairsFullConvolution::SetToTestMode(){
    if (testMode==1){
        cout<<"Stairs Convolution is already in test mode"<<endl;
        return;
    }
    if (inputActivity->dropping && primalWeight){
       kernel->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool StairsFullConvolution::HasWeightsDependency(){
    return 1;
}

bool StairsFullConvolution::NeedsUnification(){
    return 0;
}

void StairsFullConvolution::Unify(computationalNode * primalCN){
}

void StairsFullConvolution::WriteStructuredWeightsToFile(){
}

StairsFullConvolution::~StairsFullConvolution(){
}








StairsSequentialConvolution::StairsSequentialConvolution
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), numStairs(numStairs_),
    numStairConvolutions(numStairConvolutions_), symmetryLevel(symmetryLevel_){
}

void StairsSequentialConvolution::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    int vertLen = startDepth * numStairConvolutions;
    for(int stair=1; stair<numStairs; ++stair){
        vertLen += 2 * sqr(numStairConvolutions);
    }

    int nConvolutions = numStairs * numStairConvolutions;

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    indices = new int [vertLen];
    int * indices_conv = indices;
    for(int conv=0; conv<numStairConvolutions; ++conv){
        for(int j=0; j<startDepth; ++j)
            indices_conv[j] = j;
        indices_conv += startDepth;
    }

    for(int stair = 1; stair<numStairs; ++stair)
        for(int conv = 0; conv<numStairConvolutions; ++conv){
            for(int j = 0; j < 2 * numStairConvolutions; ++j)
                indices_conv[j] = j + startDepth + 2 * numStairConvolutions * (stair - 1);
            indices_conv += 2 * numStairConvolutions;
        }


    if (symmetryLevel<0 || symmetryLevel>4)
        cout<<"Symmetry level is not implemented yet";

    if (from != to ||
        kernel_hor->depth!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error0 in StairsSequentialConvolution from "<<from<<" to "<<to<<endl;

    if ((symmetryLevel == 0 && (kernel_hor->rows != 3 || kernel_hor->cols != 3) ) ||
        (symmetryLevel == 1 && (kernel_hor->rows != 3 || kernel_hor->cols != 2) ) ||
        (symmetryLevel == 2 && (kernel_hor->rows != 2 || kernel_hor->cols != 2) ) ||
        (symmetryLevel == 3 && (kernel_hor->rows != 1 || kernel_hor->cols != 3) ) ||
        (symmetryLevel == 4 && (kernel_hor->rows != 1 || kernel_hor->cols != 2) ))
        cout<<"Error1 in StairsSequentialConvolution from "<<from<<" to "<<to<<endl;

}

void StairsSequentialConvolution::ForwardPass(){
    ForwardStairsSequentialConvolution(input, kernel_vert, kernel_hor, verticalConv, startDepth, numStairs, numStairConvolutions,
                             inputActivity, indices, testMode, symmetryLevel);
}

void StairsSequentialConvolution::BackwardPass(bool computeDelta, int trueClass){
    BackwardStairsSequentialConvolution(input, inputDelta, kernel_vert, kernelGrad_vert, kernel_hor, kernelGrad_hor, verticalConv, verticalConvDelta,
                            startDepth, numStairs, numStairConvolutions, inputActivity, indices, symmetryLevel);
}

void StairsSequentialConvolution::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Stairs Convolution is already in train mode"<<endl;
        return;
    }

    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void StairsSequentialConvolution::SetToTestMode(){
    if (testMode==1){
        cout<<"Stairs Convolution is already in test mode"<<endl;
        return;
    }
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool StairsSequentialConvolution::HasWeightsDependency(){
    return 1;
}

bool StairsSequentialConvolution::NeedsUnification(){
    return 0;
}

void StairsSequentialConvolution::Unify(computationalNode * primalCN){
}

void StairsSequentialConvolution::WriteStructuredWeightsToFile(){
}

StairsSequentialConvolution::~StairsSequentialConvolution(){
    delete verticalConv;
    delete verticalConvDelta;
    delete []indices;
}













StairsPyramidalConvolution::StairsPyramidalConvolution
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_), startDepth(startDepth_), numStairs(numStairs_),
    numStairConvolutions(numStairConvolutions_){
}

void StairsPyramidalConvolution::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + 2 * numStairConvolutions * stair) * numStairConvolutions;
    }

    int nConvolutions = numStairs * numStairConvolutions;

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    indices = new int [vertLen];
    int * indices_conv = indices;
    for(int stair=0; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            for(int j=0; j<startDepth + 2 * numStairConvolutions * stair; ++j)
                indices_conv[j] = j;
            indices_conv += startDepth + 2 * numStairConvolutions * stair;
        }

    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=2||
        kernel_hor->depth!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        input->depth != startDepth + 2 * nConvolutions ||
        input->rows <= 2 * numStairs ||
        input->cols <= 2 * numStairs)
        cout<<"Error in StairsPyramidalConvolution from "<<from<<" to "<<to<<endl;
}

void StairsPyramidalConvolution::ForwardPass(){
    ForwardStairsPyramidalConvolution(input, kernel_vert, kernel_hor, verticalConv, startDepth, numStairs, numStairConvolutions,
                             inputActivity, indices, testMode);
}

void StairsPyramidalConvolution::BackwardPass(bool computeDelta, int trueClass){
    BackwardStairsPyramidalConvolution(input, inputDelta, kernel_vert, kernelGrad_vert, kernel_hor, kernelGrad_hor, verticalConv, verticalConvDelta,
                            startDepth, numStairs, numStairConvolutions, inputActivity, indices);
}

void StairsPyramidalConvolution::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Stairs Pyramidal Convolution is already in train mode"<<endl;
        return;
    }

    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void StairsPyramidalConvolution::SetToTestMode(){
    if (testMode==1){
        cout<<"Stairs Pyramidal Convolution is already in test mode"<<endl;
        return;
    }
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool StairsPyramidalConvolution::HasWeightsDependency(){
    return 1;
}

bool StairsPyramidalConvolution::NeedsUnification(){
    return 0;
}

void StairsPyramidalConvolution::Unify(computationalNode * primalCN){
}

void StairsPyramidalConvolution::WriteStructuredWeightsToFile(){
}

StairsPyramidalConvolution::~StairsPyramidalConvolution(){
    delete verticalConv;
    delete verticalConvDelta;
    delete []indices;
}








SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_),
    startDepth(startDepth_), nConvolutions(nConvolutions_), limitDepth(limitDepth_), alwaysPresentDepth(alwaysPresentDepth_){
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);

    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);

    indices = new int[vertLen];

    int * indices_j = indices;
    int trueAdditionalLen;
    int * pairIndices = new int[limitDepth / 2];

    for(int j=0; j<nConvolutions; ++j){
        for(int k=0; k<alwaysPresentDepth; ++k)
            indices_j[k] = k;
        indices_j += alwaysPresentDepth;
        trueAdditionalLen = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        FillRandom(pairIndices, (alwaysPresentDepth + 1) / 2, (startDepth + 2 * j + 1) / 2, trueAdditionalLen / 2);
        for(int k=0; k<trueAdditionalLen / 2; ++k){
            indices_j[2 * k]        = 2 * pairIndices[k] - 1;
            indices_j[2 * k + 1]    = 2 * pairIndices[k];
        }
        indices_j += trueAdditionalLen;
    }

    if (from != to ||
        kernel_hor->rows!=1||
        kernel_hor->cols!=2||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        startDepth < alwaysPresentDepth ||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric from "<<from<<" to "<<to<<endl;

    delete [] pairIndices;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::ForwardPass(){
    SequentialBottleneckConvolutionStandardRandomFullySymmetric(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions,
                                                   limitDepth, alwaysPresentDepth, inputActivity, indices, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialBottleneckConvolutionStandardRandomFullySymmetric (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                            kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                            verticalConvDelta, startDepth, nConvolutions, limitDepth, alwaysPresentDepth, inputActivity, indices);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::HasWeightsDependency(){
    return 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::NeedsUnification(){
    return 1;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::Unify(computationalNode * primalCN){
    SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric * primalNode = static_cast<SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric *> (primalCN);
    for(int j=0; j<kernel_vert->depth; ++j)
        indices[j] = primalNode->indices[j];
}

SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric::~SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric(){
    delete verticalConv;
    delete verticalConvDelta;
    delete []indices;
}








SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited
(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_):
    weightsNum_vertical(weightsNum_vertical_), weightsNum_horizontal(weightsNum_horizontal_),
    startDepth(startDepth_), nConvolutions(nConvolutions_), limitDepth(limitDepth_), alwaysPresentDepth(alwaysPresentDepth_){
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::Initiate(layers* layersData, layers* deltas, weights* weightsData,
                                weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel_vert=static_cast<tensor *>(weightsData->weightList[weightsNum_vertical].dataWeight);
    kernelGrad_vert=static_cast<tensor *>(gradient->weightList[weightsNum_vertical].dataWeight);

    kernel_hor=static_cast<tensor *>(weightsData->weightList[weightsNum_horizontal].dataWeight);
    kernelGrad_hor=static_cast<tensor *>(gradient->weightList[weightsNum_horizontal].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias_hor=static_cast<vect *>(weightsData->weightList[weightsNum_horizontal].bias);
    biasGrad_hor=static_cast<vect *>(gradient->weightList[weightsNum_horizontal].bias);

    verticalConv = new tensor(nConvolutions, input->rows, input->cols);
    verticalConvDelta = new tensor(nConvolutions, input->rows, input->cols);



    int vertLen = (alwaysPresentDepth + limitDepth) * nConvolutions;


    //cout<<"Check vertLen: "<<vertLen<<" nCOnv: "<<nConvolutions<<endl;
    if (from != to ||
        kernel_hor->rows!=3||
        kernel_hor->cols!=3||
        kernel_hor->depth!=nConvolutions||
        bias_hor->len!=nConvolutions||
        kernel_vert->rows!=1||
        kernel_vert->cols!=1||
        vertLen != kernel_vert->depth ||
        startDepth < alwaysPresentDepth + limitDepth||
        input->depth != startDepth + 2 * nConvolutions )
        cout<<"Error in SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited from "<<from<<" to "<<to<<endl;

    indices = new int[vertLen];
    int * indices_j = indices;
    int * pairIndices = new int[limitDepth / 2];

    for(int j=0; j<nConvolutions; ++j){
        for(int k=0; k<alwaysPresentDepth; ++k)
            indices_j[k] = k;
        indices_j += alwaysPresentDepth;
        FillRandom(pairIndices, (alwaysPresentDepth + 1) / 2, (startDepth + 1) / 2, limitDepth / 2);
        for(int k=0; k<limitDepth / 2; ++k){
            indices_j[2 * k]        = 2 * pairIndices[k] - 1;
            indices_j[2 * k + 1]    = 2 * pairIndices[k];
        }
        indices_j += limitDepth;
    }

//    int len=0, accLen=0;
//    if (from==0){
//        cout<<"Layer "<<from<<" structure: "<<endl;
//        for(int j=0; j<nConvolutions; ++j){
//            len = alwaysPresentDepth + limitDepth;
//            for(int k = accLen; k<accLen + len; ++k)
//                cout<<indices[k]<<" ";
//            cout<<endl;
//            accLen+=len;
//        }
//    }



    delete [] pairIndices;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::ForwardPass(){
    SequentialBottleneckConvolutionStandardRandomLimited(input, kernel_vert, kernel_hor, bias_hor, verticalConv, startDepth, nConvolutions,
                                                   limitDepth, alwaysPresentDepth, inputActivity, indices, testMode);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::BackwardPass(bool computeDelta, int trueClass){
    BackwardSequentialBottleneckConvolutionStandardRandomLimited (input, inputDelta, kernel_vert, kernelGrad_vert,
                                                            kernel_hor, kernelGrad_hor, biasGrad_hor, verticalConv,
                                                            verticalConvDelta, startDepth, nConvolutions, limitDepth, alwaysPresentDepth, inputActivity, indices);
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::SetToTrainingMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0/(1.0-inputActivity->dropRate));
    }
    testMode = 0;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::SetToTestMode(){
    if (inputActivity->dropping && primalWeight){
       kernel_vert->Multiply(1.0-inputActivity->dropRate);
    }
    testMode = 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::HasWeightsDependency(){
    return 1;
}

bool SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::NeedsUnification(){
    return 1;
}

void SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::Unify(computationalNode * primalCN){
    SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited * primalNode = static_cast<SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited *> (primalCN);
    for(int j=0; j<kernel_vert->depth; ++j)
        indices[j] = primalNode->indices[j];
}


SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited::~SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited(){
    delete verticalConv;
    delete verticalConvDelta;
    delete []indices;
}












SequentiallyConvoluteMultipleMaxMinStandard::SequentiallyConvoluteMultipleMaxMinStandard(int weightsNum_, int startDepth_, int nConvolutions_):
    weightsNum(weightsNum_), startDepth(startDepth_), nConvolutions(nConvolutions_){
}

void SequentiallyConvoluteMultipleMaxMinStandard::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor *>(gradient->weightList[weightsNum].dataWeight);

    input=static_cast<tensor*>(layersData->layerList[from]);
    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    inputActivity  = layersActivity->layerList[from];

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    if (from != to ||
        kernel->rows!=3||
        kernel->cols!=3||
        startDepth * (power(3, nConvolutions) -1) / 2 != kernel->depth ||
        bias->len != kernel->depth ||
        input->depth != startDepth * power(3, nConvolutions) )
        cout<<"Error in SequentiallyConvoluteMultipleMaxMin from "<<from<<" to "<<to<<endl;
}

void SequentiallyConvoluteMultipleMaxMinStandard::ForwardPass(){
    SequentialConvolutionMultipleStandard(input, kernel, bias, startDepth, nConvolutions, inputActivity);
}

void SequentiallyConvoluteMultipleMaxMinStandard::BackwardPass(bool computeDelta, int trueClass){
   BackwardSequentialConvolutionMultipleStandard(input, inputDelta, kernel, kernelGrad, biasGrad, startDepth, nConvolutions, computeDelta, inputActivity);
}

bool SequentiallyConvoluteMultipleMaxMinStandard::HasWeightsDependency(){
    return 1;
}


SequentiallyConvoluteMultipleMaxMinStandard::~SequentiallyConvoluteMultipleMaxMinStandard(){
}









ConvoluteDependentMaxMinMerge::ConvoluteDependentMaxMinMerge(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}


void ConvoluteDependentMaxMinMerge::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    FCkernel = static_cast<matrix *>(weightsData->weightList[weightsNum].dataWeight);
    FCbias = static_cast<vect *>(weightsData->weightList[weightsNum].bias);

    FCkernelGrad = static_cast<matrix *>(gradient->weightList[weightsNum].dataWeight);
    FCbiasGrad = static_cast<vect *>(gradient->weightList[weightsNum].bias);

    int reductionLen = FCkernel->cols;
    poolingKernelR = sqrt(input->len / reductionLen);
    poolingKernelC = poolingKernelR;

    cout<<"pooling Sizes: "<<poolingKernelR<<" "<<poolingKernelC<<endl;

    pooledInput = new tensor(input->depth, input->rows / poolingKernelR, input->cols / poolingKernelC);
    pooledInputDelta = new tensor(input->depth, input->rows / poolingKernelR, input->cols / poolingKernelC);

    int kN = (output->depth - input->depth) / 2;
    int kD = input->depth;
    int kR = input->rows + 2*paddingR + 1 - output->rows;
    int kC = input->cols + 2*paddingC + 1 - output->cols;

    cout<<"kernel sizes: "<<kN<<" "<<kD<<" "<<kR<<" "<<kC<<endl;

    convolutionKernel         = new tensor4D(kN, kD, kR, kC);
    convolutionReversedKernel = new tensor4D(kN, kD, kR, kC);
    convolutionBias = new vect(kN);

    convolutionKernelGrad = new tensor4D(kN, kD, kR, kC);
    convolutionBiasGrad = new vect(kN);


    convolutionOutput = new tensor();
    minOutput = new tensor();

    convolutionOutputDelta = new tensor();
    minOutputDelta = new tensor();

    FCkernelCkernel = new matrix();
    FCkernelCkernelGrad = new matrix();

    FCkernelCbias = new matrix();
    FCkernelCbiasGrad = new matrix();

    FCbiasCkernel = new vect();
    FCbiasCkernelGrad = new vect();

    FCbiasCbias = new vect();
    FCbiasCbiasGrad = new vect();

    indexInputRow.reserve(input->cols);
    indexOutputRow.reserve(output->cols);

    indexPooledInput.reserve(FCkernel->cols);
    indexConvolutionKernel.reserve(convolutionKernel->len);
    indexConvolutionBias.reserve(convolutionBias->len);

    cout<<"Sizes of convolution parameters"<<convolutionKernel->len<<" "<<convolutionBias->len<<endl;

    inputAddonDelta = new tensor(input->depth, input->rows, input->cols);



    if (input->rows + 2*paddingR - convolutionKernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - convolutionKernel->cols + 1 != output->cols ||
        convolutionKernel->depth != input->depth ||
        2*convolutionKernel->number + input->depth != output->depth ||
        convolutionBias->len != convolutionKernel->number ||

        pooledInput->depth != input->depth ||
        pooledInput->rows * poolingKernelR != input->rows ||
        pooledInput->cols * poolingKernelC != input->cols ||

        FCkernel->rows != convolutionKernel->len + convolutionBias->len ||
        FCkernel->cols != pooledInput->len ||
        FCbias->len != convolutionKernel->len + convolutionBias->len)
        cout<<"Error in ConvoluteDependentMaxMin from "<<from<<" to "<<to<<endl;
}

void ConvoluteDependentMaxMinMerge::ForwardPass(){
    pooledInput->SetToZero();
    AveragePool3D(input, pooledInput, poolingKernelR, poolingKernelC);

    convolutionKernel->SetToZero();
    convolutionBias->SetToZero();

    pooledInput->ListNonzeroElements(indexPooledInput);
    convolutionKernel->ListAll(indexConvolutionKernel);
    convolutionBias->ListAll(indexConvolutionBias);

    FCkernelCkernel->SubMatrix(FCkernel, convolutionKernel->len);
    FCkernelCbias->SubMatrix(FCkernel, convolutionKernel->len, convolutionBias->len);

    FCbiasCkernel->SubVect(FCbias, convolutionKernel->len);
    FCbiasCbias->SubVect(FCbias, convolutionKernel->len, convolutionBias->len);

    convolutionKernel->AddMatrVectProductBias(FCkernelCkernel, pooledInput, FCbiasCkernel, indexPooledInput, indexConvolutionKernel);
    convolutionBias->AddMatrVectProductBias(FCkernelCbias, pooledInput, FCbiasCbias, indexPooledInput, indexConvolutionBias);



    convolutionOutput -> SubTensor(output, convolutionKernel->number);
    minOutput -> SubTensor(output, convolutionKernel->number, convolutionKernel->number);
    convolutionReversedKernel->Reverse(convolutionKernel);
    Convolute3D3D(input, convolutionOutput, convolutionReversedKernel, convolutionBias, paddingR, paddingC, indexInputRow);
    minOutput->SetToMinReluFunction(convolutionOutput);
    convolutionOutput->SetToReluFunction();
    int startingIndex = 2 * convolutionOutput->len;
    output->AddThisStartingFrom(startingIndex, input);

    output->SetDroppedElementsToZero(outputActivity);
}

void ConvoluteDependentMaxMinMerge::BackwardPass(bool computeDelta, int trueClass){
    convolutionOutputDelta -> SubTensor(outputDelta, convolutionKernel->number);
    minOutputDelta->SubTensor(outputDelta, convolutionKernel->number, convolutionKernel->number);
    int startingIndex = convolutionOutputDelta->len;

    double* convOutDelta_elem = convolutionOutputDelta->elem;
    double *out_elem = output->elem;
    double* minOutDelta_elem = minOutputDelta->elem;
    double* out_elem_start = out_elem + startingIndex;

    for(int j=0; j<convolutionOutputDelta->len; ++j)
        convOutDelta_elem[j] *= (out_elem[j]>0);

    for(int j=0; j<minOutputDelta->len; ++j){
        minOutDelta_elem[j] *= (out_elem_start[j]<0);
        convOutDelta_elem[j] += minOutDelta_elem[j];
    }

    convolutionKernelGrad->SetToZero();
    convolutionBiasGrad->SetToZero();

    FCkernelCkernelGrad->SubMatrix(FCkernelGrad, convolutionKernel->len);
    FCkernelCbiasGrad->SubMatrix(FCkernelGrad, convolutionKernel->len, convolutionBias->len);

    FCbiasCkernelGrad->SubVect(FCbiasGrad, convolutionKernel->len);
    FCbiasCbiasGrad->SubVect(FCbiasGrad, convolutionKernel->len, convolutionBias->len);

    convolutionKernel->ListNonzeroElements(indexConvolutionKernel);
    convolutionBias->ListNonzeroElements(indexConvolutionBias);

    if (computeDelta){

        BackwardConvolute3D3D(input, inputDelta, convolutionOutputDelta, convolutionKernel, convolutionKernelGrad, convolutionBiasGrad, paddingR, paddingC, indexOutputRow);
        inputDelta->AddAddonStartingFrom(2*startingIndex, outputDelta);

        pooledInputDelta->SetToZero();

        pooledInputDelta->BackwardFullyConnected(FCkernelCkernel, convolutionKernelGrad, pooledInput, FCkernelCkernelGrad, FCbiasCkernelGrad, indexPooledInput, indexConvolutionKernel);
        pooledInputDelta->BackwardFullyConnected(FCkernelCbias, convolutionBiasGrad, pooledInput, FCkernelCbiasGrad, FCbiasCbiasGrad, indexPooledInput, indexConvolutionBias);

        inputAddonDelta->SetToZero();
        BackwardAveragePool3D(inputAddonDelta, pooledInputDelta, poolingKernelR, poolingKernelC);
        inputDelta->Add(inputAddonDelta);

        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else{
        BackwardConvoluteGrad3D3D(input, convolutionOutputDelta, convolutionKernelGrad, convolutionBiasGrad, paddingR, paddingC, indexOutputRow);
        FCkernelCkernelGrad ->BackwardFullyConnectedOnlyGrad(convolutionKernelGrad, pooledInput, FCbiasCkernelGrad, indexConvolutionKernel, indexPooledInput);
        FCkernelCbiasGrad   ->BackwardFullyConnectedOnlyGrad(convolutionBiasGrad, pooledInput, FCbiasCbiasGrad, indexConvolutionBias, indexPooledInput);
    }

}

bool ConvoluteDependentMaxMinMerge::HasWeightsDependency(){
    return 1;
}


ConvoluteDependentMaxMinMerge::~ConvoluteDependentMaxMinMerge(){
    delete pooledInput;
    delete pooledInputDelta;
    delete convolutionKernel;
    delete convolutionKernelGrad;
    delete convolutionReversedKernel;
    delete convolutionBias;
    delete convolutionBiasGrad;
    delete inputAddonDelta;

    DeleteOnlyShell(FCkernelCkernel);
    DeleteOnlyShell(FCkernelCkernelGrad);
    DeleteOnlyShell(FCkernelCbias);
    DeleteOnlyShell(FCkernelCbiasGrad);
    DeleteOnlyShell(FCbiasCkernel);
    DeleteOnlyShell(FCbiasCkernelGrad);
    DeleteOnlyShell(FCbiasCbias);
    DeleteOnlyShell(FCbiasCbiasGrad);
    DeleteOnlyShell(convolutionOutput);
    DeleteOnlyShell(minOutput);
    DeleteOnlyShell(convolutionOutputDelta);
    DeleteOnlyShell(minOutputDelta);
}






ConvoluteQuadraticMaxMinMerge::ConvoluteQuadraticMaxMinMerge(int weightsNum_, int paddingR_, int paddingC_):
    weightsNum(weightsNum_), paddingR(paddingR_), paddingC(paddingC_){
}

void ConvoluteQuadraticMaxMinMerge::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=static_cast<tensor4D *>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<tensor4D *>(gradient->weightList[weightsNum].dataWeight);
    reversedKernel = new tensor4D(kernel->number, kernel->depth, kernel->rows, kernel->cols);

    bias=static_cast<vect *>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect *>(gradient->weightList[weightsNum].bias);

    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    int kN_2 = kernel->number / 2;

    convolutionOutput = new tensor();
    convolutionOutput -> SubTensor(output, kN_2);

    minOutput = new tensor();
    minOutput->SubTensor(output, kN_2, kN_2);

    convolutionOutputDelta = new tensor();
    convolutionOutputDelta -> SubTensor(outputDelta, kN_2);

    minOutputDelta = new tensor();
    minOutputDelta->SubTensor(outputDelta, kN_2, kN_2);

    indexInputRow.reserve(input->cols);
    indexOutputRow.reserve(output->cols);

    startingIndexMerge = 2 * convolutionOutput->len;

    linearKernel = new tensor4D();
    linearKernel->Sub4DTensor(kernel, kN_2);

    linearKernelGrad = new tensor4D();
    linearKernelGrad->Sub4DTensor(kernelGrad, kN_2);

    linearReversedKernel = new tensor4D();
    linearReversedKernel->Sub4DTensor(reversedKernel, kN_2);

    quadraticKernel = new tensor4D();
    quadraticKernel->Sub4DTensor(kernel, kN_2, kN_2);

    quadraticKernelGrad = new tensor4D();
    quadraticKernelGrad->Sub4DTensor(kernelGrad, kN_2, kN_2);

    quadraticReversedKernel = new tensor4D();
    quadraticReversedKernel->Sub4DTensor(reversedKernel, kN_2, kN_2);

    if (input->rows + 2*paddingR - kernel->rows + 1 != output->rows ||
        input->cols + 2*paddingC - kernel->cols + 1 != output->cols ||
        kernel->depth != input->depth ||
        kernel->number + input->depth != output->depth ||
        2*bias->len != kernel->number)
        cout<<"Error in ConvoluteMaxMin from "<<from<<" to "<<to<<endl;
}



void ConvoluteQuadraticMaxMinMerge::ForwardPass(){
    reversedKernel->Reverse(kernel);
    ConvoluteQuadratic3D3D(input, convolutionOutput, linearReversedKernel, quadraticReversedKernel, bias, paddingR, paddingC, indexInputRow);

    minOutput->SetToMinReluFunction(convolutionOutput);
    convolutionOutput->SetToReluFunction();
    output->AddThisStartingFrom(startingIndexMerge, input);

    output->SetDroppedElementsToZero(outputActivity);
}

void ConvoluteQuadraticMaxMinMerge::BackwardPass(bool computeDelta, int trueClass){
    convolutionOutputDelta->MaxMinBackward(minOutputDelta, output);

    if (computeDelta){
        BackwardConvoluteQuadratic3D3D(input, inputDelta, convolutionOutputDelta, linearKernel, quadraticKernel,
                                       linearKernelGrad, quadraticKernelGrad, biasGrad, paddingR, paddingC, indexInputRow);
        inputDelta->AddAddonStartingFrom(startingIndexMerge, outputDelta);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else
        BackwardConvoluteQuadraticGrad3D3D(input, convolutionOutputDelta, linearKernelGrad, quadraticKernelGrad, biasGrad, paddingR, paddingC, indexOutputRow);
}

bool ConvoluteQuadraticMaxMinMerge::HasWeightsDependency(){
    return 1;
}


ConvoluteQuadraticMaxMinMerge::~ConvoluteQuadraticMaxMinMerge(){
    delete reversedKernel;
    DeleteOnlyShell(convolutionOutput);
    DeleteOnlyShell(minOutput);
    DeleteOnlyShell(convolutionOutputDelta);
    DeleteOnlyShell(minOutputDelta);
    DeleteOnlyShell(linearKernel);
    DeleteOnlyShell(linearKernelGrad);
    DeleteOnlyShell(linearReversedKernel);
    DeleteOnlyShell(quadraticKernel);
    DeleteOnlyShell(quadraticKernelGrad);
    DeleteOnlyShell(quadraticReversedKernel);
}








MaxPooling2D::MaxPooling2D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void MaxPooling2D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<matrix*>(layersData->layerList[from]);
    output=static_cast<matrix*>(layersData->layerList[to]);

    inputDelta=static_cast<matrix*>(deltas->layerList[from]);
    outputDelta=static_cast<matrix*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[output->len];
    colInd = new int[output->len];

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols)
        cout<<"Error in MaxPooling2D from "<<from<<" to "<<to<<endl;
}

void MaxPooling2D::ForwardPass(){
    MaxPool2D(input,output,rowInd, colInd, kernelRsize, kernelCsize);
    output->SetDroppedElementsToZero(outputActivity);
}

void MaxPooling2D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    BackwardMaxPool2D(inputDelta, outputDelta, rowInd, colInd);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxPooling2D::HasWeightsDependency(){
    return 0;
}


MaxPooling2D::~MaxPooling2D(){
    delete [] rowInd;
    delete [] colInd;
}




MaxPooling3D::MaxPooling3D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void MaxPooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[output->len];
    colInd = new int[output->len];

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols||
        output->depth != input->depth)
        cout<<"Error in MaxPooling3D from "<<from<<" to "<<to<<endl;
}

void MaxPooling3D::ForwardPass(){
    MaxPool3D(input,output,rowInd, colInd, kernelRsize, kernelCsize);
    output->SetDroppedElementsToZero(outputActivity);
}

void MaxPooling3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;

    BackwardMaxPool3D(inputDelta, outputDelta, rowInd, colInd);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxPooling3D::HasWeightsDependency(){
    return 0;
}

MaxPooling3D::~MaxPooling3D(){
    delete [] rowInd;
    delete [] colInd;
}




MaxAbsPooling3D::MaxAbsPooling3D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void MaxAbsPooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[output->len];
    colInd = new int[output->len];

    maxAbs = new tensor(output->depth, output->rows, output->cols);

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols||
        output->depth != input->depth)
        cout<<"Error in MaxAbsPooling3D from "<<from<<" to "<<to<<endl;
}

void MaxAbsPooling3D::ForwardPass(){
    MaxAbsPool3D(input, output, maxAbs, rowInd, colInd, kernelRsize, kernelCsize);
    output->SetDroppedElementsToZero(outputActivity);
}

void MaxAbsPooling3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;

    BackwardMaxAbsPool3D(inputDelta, outputDelta, rowInd, colInd);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxAbsPooling3D::HasWeightsDependency(){
    return 0;
}

MaxAbsPooling3D::~MaxAbsPooling3D(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;
}






PartialMaxAbsPooling3D::PartialMaxAbsPooling3D(int lastLayers_, int kernelRsize_, int kernelCsize_):
    lastLayers(lastLayers_), kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void PartialMaxAbsPooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    startingDepth = input->depth - lastLayers;

    partialInput = new tensor();
    partialInput->SubTensor(input, startingDepth, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, startingDepth, lastLayers);

    partialOutput = new tensor();
    partialOutput->SubTensor(output, lastLayers);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, lastLayers);

    rowInd = new int[partialOutput->len];
    colInd = new int[partialOutput->len];

    maxAbs = new tensor(partialOutput->depth, partialOutput->rows, partialOutput->cols);

    if (partialOutput->rows * kernelRsize != partialInput->rows ||
        partialOutput->cols * kernelCsize != partialInput->cols||
        output->depth < input->depth - startingDepth)
        cout<<"Error in PartialMaxAbsPooling3D from "<<from<<" to "<<to<<endl;
}

void PartialMaxAbsPooling3D::ForwardPass(){
    MaxAbsPool3D(partialInput, partialOutput, maxAbs, rowInd, colInd, kernelRsize, kernelCsize);
    partialOutput->SetDroppedElementsToZero(outputActivity, partialOutput->len);
}

void PartialMaxAbsPooling3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;

    BackwardMaxAbsPool3D(partialInputDelta, partialOutputDelta, rowInd, colInd);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startingDepth), partialInput->len);
}

bool PartialMaxAbsPooling3D::HasWeightsDependency(){
    return 0;
}


PartialMaxAbsPooling3D::~PartialMaxAbsPooling3D(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;

    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}







FullMaxAbsPooling::FullMaxAbsPooling(){
}

void FullMaxAbsPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    numLayers = input->depth;
    kernelRsize = input->rows / output->rows;
    kernelCsize = input->cols / output->cols;

    partialOutput = new tensor();
    partialOutput->SubTensor(output, numLayers);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, numLayers);

    rowInd = new int[partialOutput->len];
    colInd = new int[partialOutput->len];

    maxAbs = new tensor(partialOutput->depth, partialOutput->rows, partialOutput->cols);

    if (partialOutput->rows * kernelRsize != input->rows ||
        partialOutput->cols * kernelCsize != input->cols||
        output->depth < input->depth)
        cout<<"Error in FullMaxAbsPooling from "<<from<<" to "<<to<<endl;
}

void FullMaxAbsPooling::ForwardPass(){
    MaxAbsPool3D(input, partialOutput, maxAbs, rowInd, colInd, kernelRsize, kernelCsize);
    partialOutput->SetDroppedElementsToZero(outputActivity, partialOutput->len);
}

void FullMaxAbsPooling::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;

    BackwardMaxAbsPool3D(inputDelta, partialOutputDelta, rowInd, colInd);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool FullMaxAbsPooling::HasWeightsDependency(){
    return 0;
}

void FullMaxAbsPooling::SetToTrainingMode(){
    testMode=0;
}

void FullMaxAbsPooling::SetToTestMode(){
    testMode=1;
}


FullMaxAbsPooling::~FullMaxAbsPooling(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;

    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}




LastAveragePooling::LastAveragePooling(int lastLayers_): lastLayers(lastLayers_){
}

void LastAveragePooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    startingDepth = input->depth - lastLayers;
    kernelRsize = input->rows / output->rows;
    kernelCsize = input->cols / output->cols;

    partialInput = new tensor();
    partialInput->SubLastTensor(input, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubLastTensor(inputDelta, lastLayers);

    partialOutput = new tensor();
    partialOutput->SubTensor(output, lastLayers);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, lastLayers);

    if (partialOutput->rows * kernelRsize != partialInput->rows ||
        partialOutput->cols * kernelCsize != partialInput->cols||
        output->depth < lastLayers)
        cout<<"Error in LastAveragePooling from "<<from<<" to "<<to<<endl;
}

void LastAveragePooling::ForwardPass(){
    AveragePool3D(partialInput, partialOutput, kernelRsize, kernelCsize);

    if (!testMode)
        partialOutput->SetDroppedElementsToZero(outputActivity, partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void LastAveragePooling::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;

    BackwardAveragePool3D(partialInputDelta, partialOutputDelta, kernelRsize, kernelCsize);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startingDepth), partialInput->len);
}

void LastAveragePooling::SetToTrainingMode(){
    testMode=0;
}

void LastAveragePooling::SetToTestMode(){
    testMode=1;
}

bool LastAveragePooling::HasWeightsDependency(){
    return 0;
}


LastAveragePooling::~LastAveragePooling(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}





StructuredDropAveragePooling::StructuredDropAveragePooling(double dropRate_): dropRate(dropRate_){
}

void StructuredDropAveragePooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    lastLayers = output->depth;
    startingDepth = input->depth - lastLayers;

    activityColumns = new activityData(input->rows * input->cols, dropRate);

    partialInput = new tensor();
    partialInput->SubTensor(input, startingDepth, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, startingDepth, lastLayers);

    if (output->rows != 1 ||
        output->cols != 1 ||
        input->depth < lastLayers)
        cout<<"Error in StructuredDropAveragePooling from "<<from<<" to "<<to<<endl;
}

void StructuredDropAveragePooling::ForwardPass(){
    if (!testMode){
        activityColumns->DropUnits();
        activityLen = activityColumns->ActiveLen();
        while(activityLen == 0){
            activityColumns->DropUnits();
            activityLen = activityColumns->ActiveLen();
        }
        partialInput->SetDroppedColumnsToZero(activityColumns);
    }

    if (!testMode)
        AveragePool3D_all(partialInput, output, activityLen);

    if (testMode)
        AveragePool3D_all(partialInput, output);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity);

    if (testMode)
        if (inputActivity->dropping)
            output->Multiply(1.0 - inputActivity->dropRate);
}

void StructuredDropAveragePooling::BackwardPass(bool computeDelta, int trueClass){
    BackwardAveragePool3D_all(partialInputDelta, outputDelta, activityLen);
    partialInputDelta->SetDroppedColumnsToZero(activityColumns);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startingDepth), partialInput->len);
}

void StructuredDropAveragePooling::SetToTrainingMode(){
    testMode=0;
}

void StructuredDropAveragePooling::SetToTestMode(){
    testMode=1;
}

bool StructuredDropAveragePooling::HasWeightsDependency(){
    return 0;
}


StructuredDropAveragePooling::~StructuredDropAveragePooling(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    delete activityColumns;
}






ColumnDrop::ColumnDrop(int remainNum_): remainNum(remainNum_){
}

void ColumnDrop::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    lastLayers = output->depth;

    activityColumns = new activityData(input->rows * input->cols, 1);

    partialInput = new tensor();
    partialInput->SubLastTensor(input, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubLastTensor(inputDelta, lastLayers);

    if (output->rows != 1 ||
        output->cols != 1 ||
        input->depth < lastLayers)
        cout<<"Error in ColumnDrop from "<<from<<" to "<<to<<endl;
}

void ColumnDrop::ForwardPass(){
    if (!testMode){
        activityColumns->DropAllExcept(remainNum);
        partialInput->SetDroppedColumnsToZero(activityColumns);
    }

    if (!testMode)
        AveragePool3D_all(partialInput, output, remainNum);

    if (testMode)
        AveragePool3D_all(partialInput, output);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity);

    if (testMode)
        if (inputActivity->dropping)
            output->Multiply(1.0 - inputActivity->dropRate);
}

void ColumnDrop::BackwardPass(bool computeDelta, int trueClass){
    BackwardAveragePool3D_all(partialInputDelta, outputDelta, remainNum);
    partialInputDelta->SetDroppedColumnsToZero(activityColumns);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(input->depth - lastLayers), partialInput->len);
}

void ColumnDrop::SetToTrainingMode(){
    testMode=0;
}

void ColumnDrop::SetToTestMode(){
    testMode=1;
}

bool ColumnDrop::HasWeightsDependency(){
    return 0;
}


ColumnDrop::~ColumnDrop(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    delete activityColumns;
}






StructuredDropAverageSubPooling::StructuredDropAverageSubPooling(int border_, double dropRate_): border(border_), dropRate(dropRate_){
}

void StructuredDropAverageSubPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    lastLayers = output->depth;
    startingDepth = input->depth - lastLayers;

    activityColumns = new activityData(input->rows * input->cols, dropRate);

    partialInput = new tensor();
    partialInput->SubTensor(input, startingDepth, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, startingDepth, lastLayers);

    if (output->rows != 1 ||
        output->cols != 1||
        input->depth < lastLayers ||
        2 * border >= input->rows ||
        2 * border >= input->cols)
        cout<<"Error in PartialMaxAbsPooling3D from "<<from<<" to "<<to<<endl;
}

void StructuredDropAverageSubPooling::ForwardPass(){
    if (!testMode){
        activityColumns->DropUnits();
        partialInput->SetDroppedColumnsToZero(activityColumns);
    }

    FullSubAveragePool(partialInput, output, border);

    if (!testMode)
        output->Multiply(1.0 / (1 - dropRate));

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity);

    if (testMode)
        if (inputActivity->dropping)
            output->Multiply(1.0 - inputActivity->dropRate);
}

void StructuredDropAverageSubPooling::BackwardPass(bool computeDelta, int trueClass){
    BackwardFullSubAveragePool(partialInputDelta, outputDelta, border);
    partialInputDelta->SetDroppedColumnsToZero(activityColumns);
    partialInputDelta->Multiply(1.0 / (1 - dropRate));
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startingDepth), partialInput->len);
}

void StructuredDropAverageSubPooling::SetToTrainingMode(){
    testMode=0;
}

void StructuredDropAverageSubPooling::SetToTestMode(){
    testMode=1;
}

bool StructuredDropAverageSubPooling::HasWeightsDependency(){
    return 0;
}


StructuredDropAverageSubPooling::~StructuredDropAverageSubPooling(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    delete activityColumns;
}






PartialFirstAveragePooling3D::PartialFirstAveragePooling3D(int startingOutputDepth_, int firstLayers_, int kernelRsize_, int kernelCsize_):
    startingOutputDepth(startingOutputDepth_), firstLayers(firstLayers_), kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void PartialFirstAveragePooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialInput = new tensor();
    partialInput->SubTensor(input, firstLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, INPUT_DEPTH, firstLayers - INPUT_DEPTH);

    partialOutput = new tensor();
    partialOutput->SubTensor(output, startingOutputDepth, firstLayers);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, startingOutputDepth + INPUT_DEPTH, firstLayers - INPUT_DEPTH);

    if (partialOutput->rows * kernelRsize != partialInput->rows ||
        partialOutput->cols * kernelCsize != partialInput->cols||
        startingOutputDepth + firstLayers > output->depth ||
        firstLayers > input->depth )
        cout<<"Error in PartialFirstAveragePooling3D from "<<from<<" to "<<to<<endl;
}

void PartialFirstAveragePooling3D::ForwardPass(){
    AveragePool3D(partialInput, partialOutput, kernelRsize, kernelCsize);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void PartialFirstAveragePooling3D::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;

    BackwardAveragePool3D(partialInputDelta, partialOutputDelta, kernelRsize, kernelCsize);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void PartialFirstAveragePooling3D::SetToTrainingMode(){
    testMode=0;
}

void PartialFirstAveragePooling3D::SetToTestMode(){
    testMode=1;
}

bool PartialFirstAveragePooling3D::HasWeightsDependency(){
    return 0;
}


PartialFirstAveragePooling3D::~PartialFirstAveragePooling3D(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}





FullAveragePooling::FullAveragePooling(){
}

void FullAveragePooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    kernelRsize = input->rows / output->rows;
    kernelCsize = input->cols / output->cols;

    partialOutput = new tensor();
    partialOutput->SubTensor(output, input->depth);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, input->depth);

    if (partialOutput->rows * kernelRsize != input->rows ||
        partialOutput->cols * kernelCsize != input->cols||
        input->depth > output->depth)
        cout<<"Error in FullAveragePooling from "<<from<<" to "<<to<<endl;
}

void FullAveragePooling::ForwardPass(){
    AveragePool3D(input, partialOutput, kernelRsize, kernelCsize);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void FullAveragePooling::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;

    BackwardAveragePool3D(inputDelta, partialOutputDelta, kernelRsize, kernelCsize);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void FullAveragePooling::SetToTrainingMode(){
    if (testMode==0){
        cout<<"Full Average Pooling is already in train mode"<<endl;
        return;
    }
    testMode=0;
}

void FullAveragePooling::SetToTestMode(){
    if (testMode==1){
        cout<<"Full Average Pooling is already in test mode"<<endl;
        return;
    }

    testMode=1;
}

bool FullAveragePooling::HasWeightsDependency(){
    return 0;
}


FullAveragePooling::~FullAveragePooling(){
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}





PartialSubPooling::PartialSubPooling(int border_): border(border_){
}

void PartialSubPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    lastLayers = output->depth;

    partialInput = new tensor();
    partialInput->SubTensor(input, input->depth - lastLayers, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, input->depth - lastLayers, lastLayers);

    if (output->rows != 1 ||
        output->cols != 1 ||
        output->depth > input->depth ||
        2 * border >= input->rows ||
        2 * border >= input->cols)
        cout<<"Error in PartialSubPooling from "<<from<<" to "<<to<<endl;
}

void PartialSubPooling::ForwardPass(){
    FullSubAveragePool(partialInput, output, border);
}

void PartialSubPooling::BackwardPass(bool computeDelta, int trueClass){
    BackwardFullSubAveragePool(partialInputDelta, outputDelta, border);
}

void PartialSubPooling::SetToTrainingMode(){
    testMode=0;
}

void PartialSubPooling::SetToTestMode(){
    testMode=1;
}

bool PartialSubPooling::HasWeightsDependency(){
    return 0;
}


PartialSubPooling::~PartialSubPooling(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
}






PartialMeanVarPooling::PartialMeanVarPooling(int startingOutputDepth_): startingOutputDepth(startingOutputDepth_){
}

void PartialMeanVarPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialOutput = new tensor();
    partialOutput->SubTensor(output, startingOutputDepth,  2 * input->depth);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, startingOutputDepth, 2 * input->depth);

    if (partialOutput->rows != 1 ||
        partialOutput->cols != 1 ||
        startingOutputDepth + 2 * input->depth > output->depth)
        cout<<"Error in PartialMeanVarPooling from "<<from<<" to "<<to<<endl;
}

void PartialMeanVarPooling::ForwardPass(){
    MeanVarPoolTensor(input, partialOutput);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void PartialMeanVarPooling::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;
    BackwardMeanVarPoolTensor(input, partialOutput, inputDelta, partialOutputDelta);

    inputDelta->SetDroppedElementsToZero(inputActivity, input->len);
}

void PartialMeanVarPooling::SetToTrainingMode(){
    testMode=0;
}

void PartialMeanVarPooling::SetToTestMode(){
    testMode=1;
}

bool PartialMeanVarPooling::HasWeightsDependency(){
    return 0;
}


PartialMeanVarPooling::~PartialMeanVarPooling(){
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}




CenterPooling::CenterPooling(){
}

void CenterPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input = static_cast<tensor*>(layersData->layerList[from]);
    output = static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    sum = new vect(input->depth);

    if (output->rows != 1 ||
        output->cols != 1 ||
        output->depth != 2 * input->depth)
        cout<<"Error in CenterPooling from "<<from<<" to "<<to<<endl;
}

void CenterPooling::ForwardPass(){
    CenterPoolTensor(input, output, sum);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity);

    if (testMode)
        if (inputActivity->dropping)
            output->Multiply(1.0 - inputActivity->dropRate);
}

void CenterPooling::BackwardPass(bool computeDelta, int trueClass){
    BackwardCenterPoolTensor(output, sum, inputDelta, outputDelta);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void CenterPooling::SetToTrainingMode(){
    testMode=0;
}

void CenterPooling::SetToTestMode(){
    testMode=1;
}

bool CenterPooling::HasWeightsDependency(){
    return 0;
}

CenterPooling::~CenterPooling(){
    delete sum;
}






MedianPooling::MedianPooling(){
}

void MedianPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input = static_cast<tensor*>(layersData->layerList[from]);
    output = static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    index = new int[input->depth];

    if (output->rows != 1 ||
        output->cols != 1 ||
        output->depth != input->depth)
        cout<<"Error in MedianPooling from "<<from<<" to "<<to<<endl;
}

void MedianPooling::ForwardPass(){
    MedianNonzeroPoolTensor(input, output, index);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity);

    if (testMode)
        if (inputActivity->dropping)
            output->Multiply(1.0 - inputActivity->dropRate);
}

void MedianPooling::BackwardPass(bool computeDelta, int trueClass){
    BackwardMedianNonzeroPoolTensor(output, inputDelta, outputDelta, index);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void MedianPooling::SetToTrainingMode(){
    testMode=0;
}

void MedianPooling::SetToTestMode(){
    testMode=1;
}

bool MedianPooling::HasWeightsDependency(){
    return 0;
}

MedianPooling::~MedianPooling(){
    delete[] index;
}




QuartilesPooling::QuartilesPooling(int numQuartiles_): numQuartiles(numQuartiles_){
}

void QuartilesPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input = static_cast<tensor*>(layersData->layerList[from]);
    output = static_cast<tensor*>(layersData->layerList[to]);

    inputDelta = static_cast<tensor*>(deltas->layerList[from]);
    outputDelta = static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    index = new int[input->depth * numQuartiles];

    if (output->rows != 1 ||
        output->cols != 1 ||
        output->depth != input->depth * numQuartiles)
        cout<<"Error in QuartilesPooling from "<<from<<" to "<<to<<endl;
}

void QuartilesPooling::ForwardPass(){
    QuartilesNonzeroPoolTensor(input, output, index, numQuartiles);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity);

    if (testMode)
        if (inputActivity->dropping)
            output->Multiply(1.0 - inputActivity->dropRate);
}

void QuartilesPooling::BackwardPass(bool computeDelta, int trueClass){
    BackwardQuartilesNonzeroPoolTensor(output, inputDelta, outputDelta, index, numQuartiles);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void QuartilesPooling::SetToTrainingMode(){
    testMode=0;
}

void QuartilesPooling::SetToTestMode(){
    testMode=1;
}

bool QuartilesPooling::HasWeightsDependency(){
    return 0;
}

QuartilesPooling::~QuartilesPooling(){
    delete[] index;
}





AverageMaxAbsPooling::AverageMaxAbsPooling(int onlyAveragePoolingDepth_, int startingOutputDepth_):
    onlyAveragePoolingDepth(onlyAveragePoolingDepth_), startingOutputDepth(startingOutputDepth_){
}

void AverageMaxAbsPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    int outDepth = 2 * input->depth - onlyAveragePoolingDepth;
    int indexLen = (input->depth - onlyAveragePoolingDepth) * output->rows * output->cols;

    partialOutput = new tensor();
    partialOutput->SubTensor(output, startingOutputDepth,  outDepth);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, startingOutputDepth, outDepth);

    kernelRsize = input->rows / output->rows;
    kernelCsize = input->cols / output->cols;

    rowInd = new int[indexLen];
    colInd = new int[indexLen];

    if (partialOutput->rows * kernelRsize != input->rows ||
        partialOutput->cols * kernelCsize != input->cols ||
        startingOutputDepth + outDepth > output->depth ||
        input->depth < onlyAveragePoolingDepth)
        cout<<"Error in AverageMaxAbsPooling from "<<from<<" to "<<to<<endl;
}

void AverageMaxAbsPooling::ForwardPass(){
    AverageMaxAbsPool(input, partialOutput, kernelRsize, kernelCsize, rowInd, colInd, onlyAveragePoolingDepth);
    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void AverageMaxAbsPooling::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;
    BackwardAverageMaxAbsPool(inputDelta, partialOutputDelta, kernelRsize, kernelCsize, rowInd, colInd, onlyAveragePoolingDepth);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

void AverageMaxAbsPooling::SetToTrainingMode(){
    testMode=0;
}

void AverageMaxAbsPooling::SetToTestMode(){
    testMode=1;
}

bool AverageMaxAbsPooling::HasWeightsDependency(){
    return 0;
}


AverageMaxAbsPooling::~AverageMaxAbsPooling(){
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
    delete [] rowInd;
    delete [] colInd;
}







PartialMeanQuadStatsPooling::PartialMeanQuadStatsPooling(int startingOutputDepth_): startingOutputDepth(startingOutputDepth_){
}

void PartialMeanQuadStatsPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialOutput = new tensor();
    partialOutput->SubTensor(output, startingOutputDepth,  4 * input->depth);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, startingOutputDepth, 4 * input->depth);

    if (partialOutput->rows != 1 ||
        partialOutput->cols != 1 ||
        startingOutputDepth + 4 * input->depth > output->depth)
        cout<<"Error in PartialMeanQuadStatsPooling from "<<from<<" to "<<to<<endl;
}

void PartialMeanQuadStatsPooling::ForwardPass(){
    MeanQuadStatsPoolTensor(input, partialOutput);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void PartialMeanQuadStatsPooling::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;
    BackwardMeanQuadStatsPoolTensor(input, partialOutput, inputDelta, partialOutputDelta);

    inputDelta->SetDroppedElementsToZero(inputActivity, input->len);
}

void PartialMeanQuadStatsPooling::SetToTrainingMode(){
    testMode=0;
}

void PartialMeanQuadStatsPooling::SetToTestMode(){
    testMode=1;
}

bool PartialMeanQuadStatsPooling::HasWeightsDependency(){
    return 0;
}


PartialMeanQuadStatsPooling::~PartialMeanQuadStatsPooling(){
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}









PartialMeanStDevPooling::PartialMeanStDevPooling(int startingOutputDepth_): startingOutputDepth(startingOutputDepth_){
}

void PartialMeanStDevPooling::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialOutput = new tensor();
    partialOutput->SubTensor(output, startingOutputDepth,  2 * input->depth);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, startingOutputDepth, 2 * input->depth);

    if (partialOutput->rows != 1 ||
        partialOutput->cols != 1 ||
        startingOutputDepth + 2 * input->depth > output->depth)
        cout<<"Error in PartialMeanStDevPooling from "<<from<<" to "<<to<<endl;
}

void PartialMeanStDevPooling::ForwardPass(){
    MeanStDevPoolTensor(input, partialOutput);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void PartialMeanStDevPooling::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;
    BackwardMeanStDevPoolTensor(input, partialOutput, inputDelta, partialOutputDelta);

    inputDelta->SetDroppedElementsToZero(inputActivity, input->len);
}

void PartialMeanStDevPooling::SetToTrainingMode(){
    testMode=0;
}

void PartialMeanStDevPooling::SetToTestMode(){
    testMode=1;
}

bool PartialMeanStDevPooling::HasWeightsDependency(){
    return 0;
}


PartialMeanStDevPooling::~PartialMeanStDevPooling(){
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}









PartialFirstLastAveragePooling3D::PartialFirstLastAveragePooling3D(int startingOutputDepth_, int firstLayers_, int lastLayers_, int kernelRsize_, int kernelCsize_):
    startingOutputDepth(startingOutputDepth_), firstLayers(firstLayers_), lastLayers(lastLayers_), kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void PartialFirstLastAveragePooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialFirstInput = new tensor();
    partialFirstInput->SubTensor(input, firstLayers);

    partialFirstInputDelta = new tensor();
    partialFirstInputDelta->SubTensor(inputDelta, firstLayers);


    partialLastInput = new tensor();
    partialLastInput->SubLastTensor(input, lastLayers);

    partialLastInputDelta = new tensor();
    partialLastInputDelta->SubLastTensor(inputDelta, lastLayers);


    partialFirstOutput = new tensor();
    partialFirstOutput->SubTensor(output, startingOutputDepth, firstLayers);

    partialFirstOutputDelta = new tensor();
    partialFirstOutputDelta->SubTensor(outputDelta, startingOutputDepth, firstLayers);


    partialLastOutput = new tensor();
    partialLastOutput->SubTensor(output, startingOutputDepth + firstLayers, lastLayers);

    partialLastOutputDelta = new tensor();
    partialLastOutputDelta->SubTensor(outputDelta, startingOutputDepth + firstLayers, lastLayers);



    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols||
        startingOutputDepth + firstLayers + lastLayers > output->depth ||
        firstLayers + lastLayers > input->depth )
        cout<<"Error in PartialFirstLastAveragePooling3D from "<<from<<" to "<<to<<endl;
}

void PartialFirstLastAveragePooling3D::ForwardPass(){
    AveragePool3D(partialFirstInput, partialFirstOutput, kernelRsize, kernelCsize);
    AveragePool3D(partialLastInput, partialLastOutput, kernelRsize, kernelCsize);

    if (!testMode){
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialFirstOutput->len + partialLastOutput->len);
    }


    if (testMode)
        if (inputActivity->dropping){
            partialFirstOutput->Multiply(1.0 - inputActivity->dropRate);
            partialLastOutput->Multiply(1.0 - inputActivity->dropRate);
        }
}

void PartialFirstLastAveragePooling3D::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;
    //if (!PARALLEL_ARCHITECTURE)
    BackwardAveragePool3D(partialFirstInputDelta, partialFirstOutputDelta, kernelRsize, kernelCsize);

    BackwardAveragePool3D(partialLastInputDelta, partialLastOutputDelta, kernelRsize, kernelCsize);

    inputDelta->SetDroppedElementsToZero(inputActivity, partialFirstInput->len);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->len - partialLastInputDelta->len, partialLastInputDelta->len);
}

void PartialFirstLastAveragePooling3D::SetToTrainingMode(){
    testMode=0;
}

void PartialFirstLastAveragePooling3D::SetToTestMode(){
    testMode=1;
}

bool PartialFirstLastAveragePooling3D::HasWeightsDependency(){
    return 0;
}


PartialFirstLastAveragePooling3D::~PartialFirstLastAveragePooling3D(){
    DeleteOnlyShell(partialFirstInput);
    DeleteOnlyShell(partialFirstInputDelta);
    DeleteOnlyShell(partialFirstOutput);
    DeleteOnlyShell(partialFirstOutputDelta);

    DeleteOnlyShell(partialLastInput);
    DeleteOnlyShell(partialLastInputDelta);
    DeleteOnlyShell(partialLastOutput);
    DeleteOnlyShell(partialLastOutputDelta);
}






PartialMiddleAveragePooling3D::PartialMiddleAveragePooling3D(int startingInputDepth_, int numLayers_, int startingOutputDepth_, int kernelRsize_, int kernelCsize_):
    startingInputDepth(startingInputDepth_), numLayers(numLayers_), startingOutputDepth(startingOutputDepth_), kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void PartialMiddleAveragePooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    partialInput = new tensor();
    partialInput->SubTensor(input, startingInputDepth, numLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, startingInputDepth, numLayers);

    partialOutput = new tensor();
    partialOutput->SubTensor(output, startingOutputDepth, numLayers);

    partialOutputDelta = new tensor();
    partialOutputDelta->SubTensor(outputDelta, startingOutputDepth, numLayers);

    if (partialOutput->rows * kernelRsize != partialInput->rows ||
        partialOutput->cols * kernelCsize != partialInput->cols||
        startingInputDepth + numLayers > input->depth ||
        startingOutputDepth + numLayers > output->depth)
        cout<<"Error in PartialMiddleAveragePooling3D from "<<from<<" to "<<to<<endl;
}

void PartialMiddleAveragePooling3D::ForwardPass(){
    AveragePool3D(partialInput, partialOutput, kernelRsize, kernelCsize);

    if (!testMode)
        output->SetDroppedElementsToZero(outputActivity, output->Ind(startingOutputDepth), partialOutput->len);

    if (testMode)
        if (inputActivity->dropping)
            partialOutput->Multiply(1.0 - inputActivity->dropRate);
}

void PartialMiddleAveragePooling3D::BackwardPass(bool computeDelta, int trueClass){
    //if (!computeDelta) return;

    BackwardAveragePool3D(partialInputDelta, partialOutputDelta, kernelRsize, kernelCsize);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startingInputDepth), partialInputDelta->len);
}

void PartialMiddleAveragePooling3D::SetToTrainingMode(){
    testMode=0;
}

void PartialMiddleAveragePooling3D::SetToTestMode(){
    testMode=1;
}

bool PartialMiddleAveragePooling3D::HasWeightsDependency(){
    return 0;
}

PartialMiddleAveragePooling3D::~PartialMiddleAveragePooling3D(){
    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}












MaxMinPooling3D::MaxMinPooling3D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void MaxMinPooling3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[output->len];
    colInd = new int[output->len];

    maxOut = new tensor();
    minOut = new tensor();

    maxOutDelta = new tensor();
    minOutDelta = new tensor();

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols||
        output->depth != 2*input->depth)
        cout<<"Error in MaxPooling3D from "<<from<<" to "<<to<<endl;
}

void MaxMinPooling3D::ForwardPass(){
    maxOut->SubTensor(output, input->depth);
    minOut->SubTensor(output, input->depth, input->depth);

    int startingIndex = output->len / 2;

    MaxPool3D(input, maxOut, rowInd, colInd, kernelRsize, kernelCsize);
    MinPool3D(input, minOut, rowInd + startingIndex, colInd + startingIndex, kernelRsize, kernelCsize);

    output->SetDroppedElementsToZero(outputActivity);
}

void MaxMinPooling3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    maxOutDelta->SubTensor(outputDelta, inputDelta->depth);
    minOutDelta->SubTensor(outputDelta, inputDelta->depth, inputDelta->depth);

    int startingIndex = output->len / 2;

    BackwardMaxPool3D(inputDelta, maxOutDelta, rowInd, colInd);
    BackwardMinPool3D(inputDelta, minOutDelta, rowInd + startingIndex, colInd + startingIndex);

    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxMinPooling3D::HasWeightsDependency(){
    return 0;
}


MaxMinPooling3D::~MaxMinPooling3D(){
    delete [] rowInd;
    delete [] colInd;
    DeleteOnlyShell(maxOut);
    DeleteOnlyShell(minOut);
    DeleteOnlyShell(maxOutDelta);
    DeleteOnlyShell(minOutDelta);

}




MaxMinPoolingIndex3D::MaxMinPoolingIndex3D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void MaxMinPoolingIndex3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[output->len / 3];
    colInd = new int[output->len / 3];

    maxOut = new tensor();
    minOut = new tensor();

    maxOutDelta = new tensor();
    minOutDelta = new tensor();

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols||
        output->depth != 6*input->depth)
        cout<<"Error in MaxPoolingIndex3D from "<<from<<" to "<<to<<endl;
}

void MaxMinPoolingIndex3D::ForwardPass(){
    maxOut->SubTensor(output, input->depth);
    minOut->SubTensor(output, input->depth, input->depth);

    int startingIndex = output->len / 6;

    MaxPool3D(input, maxOut, rowInd, colInd, kernelRsize, kernelCsize);
    MinPool3D(input, minOut, rowInd + startingIndex, colInd + startingIndex, kernelRsize, kernelCsize);

    double* out_elem_start = output->elem + 2*startingIndex;
    double row_0 = rowInd[0];
    double col_0 = colInd[0];
    for(int j=0; j<output->len / 3; ++j){
        out_elem_start[j] = ( (double)rowInd[j] - row_0) / input->rows;
    }

    out_elem_start = output->elem + 4*startingIndex;
    for(int j=0; j<output->len / 3; ++j){
        out_elem_start[j] =( (double) colInd[j] - col_0) / input->cols;
    }

    output->SetDroppedElementsToZero(outputActivity);
}

void MaxMinPoolingIndex3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    maxOutDelta->SubTensor(outputDelta, inputDelta->depth);
    minOutDelta->SubTensor(outputDelta, inputDelta->depth, inputDelta->depth);

    int startingIndex = output->len / 6;

    BackwardMaxPool3D(inputDelta, maxOutDelta, rowInd, colInd);
    BackwardMinPool3D(inputDelta, minOutDelta, rowInd + startingIndex, colInd + startingIndex);

    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxMinPoolingIndex3D::HasWeightsDependency(){
    return 0;
}


MaxMinPoolingIndex3D::~MaxMinPoolingIndex3D(){
    delete [] rowInd;
    delete [] colInd;
    DeleteOnlyShell(maxOut);
    DeleteOnlyShell(minOut);
    DeleteOnlyShell(maxOutDelta);
    DeleteOnlyShell(minOutDelta);

}




MaxAbsPoolingIndex3D::MaxAbsPoolingIndex3D(int kernelRsize_, int kernelCsize_):
    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_){
}

void MaxAbsPoolingIndex3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<tensor*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<tensor*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[output->len / 3];
    colInd = new int[output->len / 3];

    maxAbs = new tensor(input->depth, output->rows, output->cols);

    if (output->rows * kernelRsize != input->rows ||
        output->cols * kernelCsize != input->cols||
        output->depth != 3 * input->depth)
        cout<<"Error in MaxAbsPoolingIndex3D from "<<from<<" to "<<to<<endl;
}

void MaxAbsPoolingIndex3D::ForwardPass(){
    MaxAbsPoolIndex3D(input, output, maxAbs, rowInd, colInd, kernelRsize, kernelCsize);
    output->SetDroppedElementsToZero(outputActivity);
}

void MaxAbsPoolingIndex3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    BackwardMaxAbsPoolIndex3D(inputDelta, outputDelta, rowInd, colInd);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxAbsPoolingIndex3D::HasWeightsDependency(){
    return 0;
}


MaxAbsPoolingIndex3D::~MaxAbsPoolingIndex3D(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;
}







MaxAbsPoolingSoftIndex3D::MaxAbsPoolingSoftIndex3D(double logFactor_): logFactor(logFactor_){
}

void MaxAbsPoolingSoftIndex3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<vect*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<vect*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[input->depth];
    colInd = new int[input->depth];
    maxAbs = new vect(input->depth);

    softMaxInput = new tensor(input->depth, input->rows, input->cols);

    if (output->len != 3 * input->depth)
        cout<<"Error in MaxAbsPoolingSoftIndex3D from "<<from<<" to "<<to<<endl;
}

void MaxAbsPoolingSoftIndex3D::ForwardPass(){
    MaxAbsPoolSoftIndex3D(input, output, maxAbs, softMaxInput, rowInd, colInd, logFactor);
    output->SetDroppedElementsToZero(outputActivity);
}

void MaxAbsPoolingSoftIndex3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    BackwardMaxAbsPoolSoftIndex3D(input, softMaxInput, output, inputDelta, outputDelta, rowInd, colInd, logFactor);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxAbsPoolingSoftIndex3D::HasWeightsDependency(){
    return 0;
}

MaxAbsPoolingSoftIndex3D::~MaxAbsPoolingSoftIndex3D(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;
    delete softMaxInput;
}




MaxAbsPoolingSoftDiffIndex3D::MaxAbsPoolingSoftDiffIndex3D(double logFactor_): logFactor(logFactor_){
}

void MaxAbsPoolingSoftDiffIndex3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<vect*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<vect*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    rowInd = new int[input->depth];
    colInd = new int[input->depth];
    maxAbs = new vect(input->depth);
    tempOutput = new vect(output->len);
    tempOutputDelta = new vect(outputDelta->len);

    softMaxInput = new tensor(input->depth, input->rows, input->cols);

    if (output->len != 3 * input->depth)
        cout<<"Error in MaxAbsPoolingSoftDiffIndex3D from "<<from<<" to "<<to<<endl;
}

void MaxAbsPoolingSoftDiffIndex3D::ForwardPass(){
    MaxAbsPoolSoftDiffIndex3D(input, output, tempOutput, maxAbs, softMaxInput, rowInd, colInd, logFactor);
    output->SetDroppedElementsToZero(outputActivity);
}

void MaxAbsPoolingSoftDiffIndex3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    BackwardMaxAbsPoolSoftDiffIndex3D(input, softMaxInput, tempOutput, inputDelta, outputDelta, tempOutputDelta, rowInd, colInd, logFactor);
    inputDelta->SetDroppedElementsToZero(inputActivity);
}

bool MaxAbsPoolingSoftDiffIndex3D::HasWeightsDependency(){
    return 0;
}

MaxAbsPoolingSoftDiffIndex3D::~MaxAbsPoolingSoftDiffIndex3D(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;
    delete softMaxInput;
    delete tempOutput;
    delete tempOutputDelta;
}





PartialMaxAbsPoolingSoftDiffIndex3D::PartialMaxAbsPoolingSoftDiffIndex3D(int lastLayers_, double logFactor_) :
    logFactor(logFactor_), lastLayers(lastLayers_){
}

void PartialMaxAbsPoolingSoftDiffIndex3D::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    input=static_cast<tensor*>(layersData->layerList[from]);
    output=static_cast<vect*>(layersData->layerList[to]);

    inputDelta=static_cast<tensor*>(deltas->layerList[from]);
    outputDelta=static_cast<vect*>(deltas->layerList[to]);

    inputActivity = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    startingDepth = input->depth - lastLayers;

    partialInput = new tensor();
    partialInput->SubTensor(input, startingDepth, lastLayers);

    partialInputDelta = new tensor();
    partialInputDelta->SubTensor(inputDelta, startingDepth, lastLayers);

    partialOutput = new vect();
    partialOutput->SubVect(output, lastLayers * 3);

    partialOutputDelta = new vect();
    partialOutputDelta->SubVect(outputDelta, lastLayers * 3);


    rowInd = new int[lastLayers];
    colInd = new int[lastLayers];
    maxAbs = new vect(lastLayers);
    tempOutput = new vect(partialOutput->len);
    tempOutputDelta = new vect(partialOutputDelta->len);

    softMaxInput = new tensor(partialInput->depth, partialInput->rows, partialInput->cols);

    if (output->len < partialOutput->len)
        cout<<"Error in MaxAbsPoolingSoftDiffIndex3D from "<<from<<" to "<<to<<endl;
}

void PartialMaxAbsPoolingSoftDiffIndex3D::ForwardPass(){
    MaxAbsPoolSoftDiffIndex3D(partialInput, partialOutput, tempOutput, maxAbs, softMaxInput, rowInd, colInd, logFactor);
    partialOutput->SetDroppedElementsToZero(outputActivity, partialOutput->len);
}

void PartialMaxAbsPoolingSoftDiffIndex3D::BackwardPass(bool computeDelta, int trueClass){
    if (!computeDelta) return;
    BackwardMaxAbsPoolSoftDiffIndex3D(partialInput, softMaxInput, tempOutput, partialInputDelta, partialOutputDelta, tempOutputDelta, rowInd, colInd, logFactor);
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startingDepth), partialInput->len);
}

bool PartialMaxAbsPoolingSoftDiffIndex3D::HasWeightsDependency(){
    return 0;
}

PartialMaxAbsPoolingSoftDiffIndex3D::~PartialMaxAbsPoolingSoftDiffIndex3D(){
    delete [] rowInd;
    delete [] colInd;
    delete maxAbs;
    delete softMaxInput;
    delete tempOutput;
    delete tempOutputDelta;

    DeleteOnlyShell(partialInput);
    DeleteOnlyShell(partialInputDelta);
    DeleteOnlyShell(partialOutput);
    DeleteOnlyShell(partialOutputDelta);
}











FCMaxMinMerge::FCMaxMinMerge(int weightsNum_): weightsNum(weightsNum_){
}

void FCMaxMinMerge::Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner){
    primalWeight = primalWeightOwner;
    kernel=    static_cast<matrix*>(weightsData->weightList[weightsNum].dataWeight);
    kernelGrad=static_cast<matrix*>(   gradient->weightList[weightsNum].dataWeight);

    bias=    static_cast<vect*>(weightsData->weightList[weightsNum].bias);
    biasGrad=static_cast<vect*>(   gradient->weightList[weightsNum].bias);

    input = new vect();
    output = new vect();
    inputDelta = new vect();
    outputDelta = new vect();

    indexInput.reserve(kernel->cols);
    indexOutput.reserve(kernel->rows);

    input->StaticCastToVect(layersData->layerList[from]);
    output->StaticCastToVect(layersData->layerList[to]);

    inputDelta->StaticCastToVect(deltas->layerList[from]);
    outputDelta->StaticCastToVect(deltas->layerList[to]);

    FCOutput = new vect();
    minOutput = new vect();

    FCOutputDelta = new vect();
    minOutputDelta = new vect();

    inputActivity  = layersActivity->layerList[from];
    outputActivity = layersActivity->layerList[to];

    if (input->len != kernel->cols ||
        output->len != 2*kernel->rows + input->len ||
        bias->len != kernel->rows)
        cout<<"Error in FcMaxMin from layer "<<from<<" to layer "<<to<<endl;

    FCOutput->SubVect(output, kernel->rows);
    minOutput->SubVect(output, kernel->rows, kernel->rows);
    FCOutputDelta->SubVect(outputDelta, kernel->rows);
    minOutputDelta->SubVect(outputDelta, kernel->rows, kernel->rows);
    startingIndexMerge = 2 * FCOutput->len;
}

void FCMaxMinMerge::ForwardPass(){
    input->ListNonzeroActiveElements(indexInput, inputActivity);
    FCOutput->ListActiveElements(indexOutput, outputActivity);

    FCOutput->AddMatrVectProductBias(kernel, input, bias, indexInput, indexOutput);
    minOutput->SetToMinReluFunction(FCOutput);
    FCOutput->SetToReluFunction();

    output->AddThisStartingFrom(startingIndexMerge, input);

    output->SetDroppedElementsToZero(outputActivity);

}

void FCMaxMinMerge::BackwardPass(bool computeDelta, int trueClass){

    FCOutputDelta->MaxMinBackward(minOutputDelta, output);
    FCOutputDelta->ListNonzeroActiveElements(indexOutput, outputActivity);

    if (computeDelta){
        inputDelta->BackwardFullyConnected(kernel, FCOutputDelta, input, kernelGrad, biasGrad, indexInput, indexOutput);

        inputDelta->AddAddonStartingFrom(startingIndexMerge, outputDelta);
        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    else
        kernelGrad->BackwardFullyConnectedOnlyGrad(FCOutputDelta, input, biasGrad, indexOutput, indexInput);
}

bool FCMaxMinMerge::HasWeightsDependency(){
    return 1;
}


FCMaxMinMerge::~FCMaxMinMerge(){
    DeleteOnlyShell(input);
    DeleteOnlyShell(output);
    DeleteOnlyShell(inputDelta);
    DeleteOnlyShell(outputDelta);
    DeleteOnlyShell(FCOutput);
    DeleteOnlyShell(minOutput);
    DeleteOnlyShell(FCOutputDelta);
    DeleteOnlyShell(minOutputDelta);
}



//MaxPooling3D::MaxPooling3D(int kernelRsize_, int kernelCsize_, int strideR_, int strideC_, int depth, int rows, int cols):
//    kernelRsize(kernelRsize_), kernelCsize(kernelCsize_), strideR(strideR_), strideC(strideC_){
//        rowInd=new int **[depth];
//        colInd=new int **[depth];
//        for(int d=0; d<depth; d++){
//            rowInd[d]=new int* [rows];
//            colInd[d]=new int* [rows];
//            for(int r=0; r<rows; r++){
//                rowInd[d][r]=new int [cols];
//                colInd[d][r]=new int [cols];
//            }
//        }
//}
//
//void MaxPooling3D::ForwardPass(int from, int to){
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    tensor* output=static_cast<tensor*>(layersData->layerList[to]);
//
//    activityTensor*  inputActivity = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityTensor* outputActivity = static_cast<activityTensor*>(layersActivity->layerList[to]);
//
//    DroppedMaxPool3D(input,output,rowInd, colInd, kernelRsize, kernelCsize, strideR, strideC,
//                     inputActivity->activeUnits, outputActivity->activeUnits);
//}
//
//void MaxPooling3D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    tensor* inputDelta=static_cast<tensor*>(deltas->layerList[from]);
//    tensor* outputDelta=static_cast<tensor*>(deltas->layerList[to]);
//
//    activityTensor*  inputActivity = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityTensor* outputActivity = static_cast<activityTensor*>(layersActivity->layerList[to]);
//
//    DroppedBackwardMaxPool3D(inputDelta, outputDelta, rowInd, colInd, inputActivity->activeUnits, outputActivity->activeUnits);
//}
//
//bool MaxPooling3D::HasWeightsDependency(){
//    return 0;
//}
//
//void MaxPooling3D::BackpropagateDroppedUnits(int from, int to){
//    activityTensor*  inputActivity = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityTensor* outputActivity = static_cast<activityTensor*>(layersActivity->layerList[to]);
//
//    int depth=inputActivity->activeUnits.size();
//    int rows=inputActivity->activeUnits[0].size();
//    int cols=inputActivity->activeUnits[0][0].size();
//
//    for(int d=0; d<depth; d++)
//        for(int iR=0, oR=0; iR<rows-kernelRsize+1; iR+=strideR, ++oR)
//            for(int iC=0, oC=0; iC<cols-kernelCsize+1; iC+=strideC, ++oC){
//                    if (outputActivity->activeUnits[d][oR][oR]) continue;
//                    for(int kR=0; kR<kernelRsize; kR++)
//                        for(int kC=0; kC<kernelCsize; kC++)
//                            inputActivity->activeUnits[d][iR+kR][iC+kC]=0;
//            }
//
//}



//TranslationInvariant::TranslationInvariant(int maxDist_): maxDist(maxDist_){
//    res1 = new matrix(maxDist+1, maxDist+1);
//    res2 = new matrix(maxDist+1, maxDist+1);
//    res3 = new matrix(maxDist+1, maxDist+1);
//}
//
//void TranslationInvariant::ForwardPass(int from, int to){
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    vect* output=static_cast<vect*>(layersData->layerList[to]);
//
//    output->elem[0]=input->Sum();
//    //matrix* res = new matrix(maxDist+1, maxDist+1);
//
//    for(int oR=0; oR<=maxDist; oR++)
//        for(int oC=0; oC<=maxDist; oC++){
//            res1->elem[oR][oC]=0;
//            res2->elem[oR][oC]=0;
//            res3->elem[oR][oC]=0;
//            for(int iR=0; iR<input->rows-oR; iR++)
//                for(int iC=0; iC<input->cols-oC; iC++){
//                    res1->elem[oR][oC]+=input->elem[iR][iC]*input->elem[iR+oR][iC+oC];
//                    res2->elem[oR][oC]+=input->elem[iR][iC]*sqr(input->elem[iR+oR][iC+oC]);
//                    res3->elem[oR][oC]+=sqr(input->elem[iR][iC])*input->elem[iR+oR][iC+oC];
//                }
//        }
//    int matrSize=(maxDist+1)*(maxDist+1);
//    for(int oR=0; oR<=maxDist; oR++)
//        for(int oC=0; oC<=maxDist; oC++){
//            output->elem[oR*(maxDist+1)+oC+1]           =res1->elem[oR][oC];
//            output->elem[oR*(maxDist+1)+oC+1+matrSize]  =res2->elem[oR][oC];
//            output->elem[oR*(maxDist+1)+oC+1+2*matrSize]=res3->elem[oR][oC];
//        }
//}
//
//void TranslationInvariant::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    matrix* inputDelta=static_cast<matrix*>(deltas->layerList[from]);
//    vect* outputDelta=static_cast<vect*>(deltas->layerList[to]);
//
//
//}
//
//bool TranslationInvariant::HasWeightsDependency(){
//    return 0;
//}


//
//CenterAtTheMean2D::CenterAtTheMean2D(){
//}
//
//void CenterAtTheMean2D::ForwardPass(int from, int to){
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    matrix* output=static_cast<matrix*>(layersData->layerList[to]);
//    double imSum = input->Sum();//-input->rows*input->cols*(minVal-1.0);
//
//    double avRow=0, avCol=0;
//    for(int r=0; r<input->rows; r++)
//        for(int c=0; c<input->cols; c++){
//            avRow+=double(r)*(input->elem[r][c]);//-minVal+1.0);
//            avCol+=double(c)*(input->elem[r][c]);//-minVal+1.0);
//        }
//    avRow/=imSum;
//    avCol/=imSum;
//
//    rowShift=lrint(avRow);
//    colShift=lrint(avCol);
//
//    for(int r=0; r<input->rows; r++)
//        for(int c=0; c<input->cols; c++){
//            output->elem[r-rowShift+input->rows-1][c-colShift+input->cols-1]+=input->elem[r][c];
//        }
//}
//
//void CenterAtTheMean2D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    matrix* inputDelta=static_cast<matrix*>(deltas->layerList[from]);
//    matrix* outputDelta=static_cast<matrix*>(deltas->layerList[to]);
//
//    for(int r=0; r<input->rows; r++)
//        for(int c=0; c<input->cols; c++){
//            inputDelta->elem[r][c]+=outputDelta->elem[r-rowShift+input->rows-1][c-colShift+input->cols-1];
//        }
//}
//
//bool CenterAtTheMean2D::HasWeightsDependency(){
//    return 0;
//}
//
//
//
//CenterAtTheMean3D::CenterAtTheMean3D(){
//}
//
//void CenterAtTheMean3D::ForwardPass(int from, int to){
//    tensor* input =static_cast<tensor*>(layersData->layerList[from]);
//    tensor* output=static_cast<tensor*>(layersData->layerList[to]);
//    double minVal = input->FindMin();
//    double imSum = input->Sum()-input->depth*input->rows*input->cols*(minVal-1.0);
//
//    double avRow=0, avCol=0;
//    for(int d=0; d<input->depth; d++)
//        for(int r=0; r<input->rows; r++)
//            for(int c=0; c<input->cols; c++){
//                avRow+=double(r)*(input->elem[d][r][c]-minVal+1.0);
//                avCol+=double(c)*(input->elem[d][r][c]-minVal+1.0);
//            }
//
//    avRow/=imSum;
//    avCol/=imSum;
//
//    rowShift=lrint(avRow);
//    colShift=lrint(avCol);
//
//    for(int d=0; d<input->depth; d++)
//        for(int r=0; r<input->rows; r++)
//            for(int c=0; c<input->cols; c++){
//                output->elem[d][r-rowShift+input->rows-1][c-colShift+input->cols-1]+=input->elem[d][r][c];
//            }
//}
//
//
//void CenterAtTheMean3D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    tensor* inputDelta=static_cast<tensor*>(deltas->layerList[from]);
//    tensor* outputDelta=static_cast<tensor*>(deltas->layerList[to]);
//    for(int d=0; d<input->depth; d++)
//        for(int r=0; r<input->rows; r++)
//            for(int c=0; c<input->cols; c++){
//                inputDelta->elem[d][r][c]+=outputDelta->elem[d][r-rowShift+input->rows-1][c-colShift+input->cols-1];
//            }
//}
//
//bool CenterAtTheMean3D::HasWeightsDependency(){
//    return 0;
//}









//
//Merge2D::Merge2D(int startingIndex_): startingIndex(startingIndex_){
//}
//
//void Merge2D::ForwardPass(int from, int to){
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    vect* output=static_cast<vect*>(layersData->layerList[to]);
//
//    activityMatrix* inputActivity  = static_cast<activityMatrix*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    double * inputElemR;
//    int rInpCols;
//
//    for(int r=0; r<input->rows; r++){
//        rInpCols = r*input->cols + startingIndex;
//        inputElemR = input->elem[r];
//        vector<bool> &inputActiveR = inputActivity->activeUnits[r];
//        for(int c=0; c<input->cols; c++)
//            if (inputActiveR[c] && outputActivity->activeUnits[rInpCols+c])
//                output->elem[rInpCols+c]+=inputElemR[c];
//    }
//}
//
//void Merge2D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    matrix* inputDelta=static_cast<matrix*>(deltas->layerList[from]);
//    vect* outputDelta=static_cast<vect*>(deltas->layerList[to]);
//
//    activityMatrix* inputActivity  = static_cast<activityMatrix*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    int rInpCols;
//    double * inputDeltaElemR;
//    for(int r=0; r<input->rows; r++){
//        rInpCols = r*input->cols+startingIndex;
//        inputDeltaElemR = inputDelta->elem[r];
//        vector<bool> &inputActiveR = inputActivity->activeUnits[r];
//        for(int c=0; c<input->cols; c++)
//            if (inputActiveR[c] && outputActivity->activeUnits[rInpCols+c])
//                inputDeltaElemR[c]+=outputDelta->elem[rInpCols+c];
//    }
//}
//
//bool Merge2D::HasWeightsDependency(){
//    return 0;
//}
//
//void Merge2D::BackpropagateDroppedUnits(int from, int to){
//    activityMatrix* inputActivity  = static_cast<activityMatrix*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    int rows = inputActivity->activeUnits.size();
//    int cols = inputActivity->activeUnits[0].size();
//
//    for(int r=0; r<rows; r++)
//        for(int c=0; c<cols; c++)
//            if (!outputActivity->activeUnits[startingIndex+r*cols+c])
//                inputActivity->activeUnits[r][c]=0;
//}
//
//
//
//
//Merge3D::Merge3D(int startingIndex_): startingIndex(startingIndex_){
//}
//
//void Merge3D::ForwardPass(int from, int to){
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    vect* output =static_cast<vect*>(layersData->layerList[to]);
//
//    activityTensor* inputActivity  = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityVect*   outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    matrix* inputElemD;
//    double * inputElemDR;
//    int dRowsCols;
//    int dRowsColsPlusRCols;
//    for(int d=0; d<input->depth; d++){
//        inputElemD = &(input->elem[d]);
//        dRowsCols=d*input->rows*input->cols;
//        vector<vector<bool> >& inputActD = inputActivity->activeUnits[d];
//
//        for(int r=0; r<input->rows; r++){
//            inputElemDR = inputElemD->elem[r];
//            dRowsColsPlusRCols = startingIndex + dRowsCols + r*input->cols;
//            vector<bool> & inputActDR = inputActD[r];
//
//            for(int c=0; c<input->cols; c++)
//                if (inputActDR[c] && outputActivity->activeUnits[dRowsColsPlusRCols+c])
//                    output->elem[dRowsColsPlusRCols+c]+=inputElemDR[c];
//        }
//
//    }
//
//}
//
//void Merge3D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    tensor* inputDelta=static_cast<tensor*>(deltas->layerList[from]);
//    vect* outputDelta=static_cast<vect*>(deltas->layerList[to]);
//
//    activityTensor* inputActivity  = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityVect*   outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    matrix* inputDeltaD;
//    double *inputDeltaDR;
//    int dRowsCols;
//    int dRowsColsPlusRCols;
//    for(int d=0; d<input->depth; d++){
//        dRowsCols=d*input->rows*input->cols;
//        inputDeltaD = &(inputDelta->elem[d]);
//        vector<vector<bool> >& inputActD = inputActivity->activeUnits[d];
//
//        for(int r=0; r<input->rows; r++){
//            dRowsColsPlusRCols = startingIndex + dRowsCols + r*input->cols;
//            inputDeltaDR = inputDeltaD->elem[r];
//            vector<bool> & inputActDR = inputActD[r];
//
//            for(int c=0; c<input->cols; c++)
//                if (inputActDR[c] && outputActivity->activeUnits[dRowsColsPlusRCols+c])
//                    inputDeltaDR[c]+=outputDelta->elem[dRowsColsPlusRCols+c];
//        }
//
//    }
//}
//
//bool Merge3D::HasWeightsDependency(){
//    return 0;
//}
//
//void Merge3D::BackpropagateDroppedUnits(int from, int to){
//    activityTensor* inputActivity  = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    int depth= inputActivity->activeUnits.size();
//    int rows = inputActivity->activeUnits[0].size();
//    int cols = inputActivity->activeUnits[0][0].size();
//
//    for(int d=0; d<depth; d++)
//        for(int r=0; r<rows; r++)
//            for(int c=0; c<cols; c++)
//                if (!outputActivity->activeUnits[startingIndex + d*rows*cols+r*cols+c])
//                    inputActivity->activeUnits[d][r][c]=0;
//}
//


//Flatten2D::Flatten2D(){
//}
//
//void Flatten2D::ForwardPass(int from, int to){
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    vect* output=static_cast<vect*>(layersData->layerList[to]);
//
//    activityMatrix* inputActivity  = static_cast<activityMatrix*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    double * inputElemR;
//    int rInpCols;
//
//    for(int r=0; r<input->rows; r++){
//        rInpCols = r*input->cols;
//        inputElemR = input->elem[r];
//        vector<bool> &inputActiveR = inputActivity->activeUnits[r];
//        for(int c=0; c<input->cols; c++)
//            if (inputActiveR[c] && outputActivity->activeUnits[rInpCols+c])
//                output->elem[rInpCols+c]+=inputElemR[c];
//    }
//}
//
//void Flatten2D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//
//    matrix* input=static_cast<matrix*>(layersData->layerList[from]);
//    matrix* inputDelta=static_cast<matrix*>(deltas->layerList[from]);
//    vect* outputDelta=static_cast<vect*>(deltas->layerList[to]);
//
//    activityMatrix* inputActivity  = static_cast<activityMatrix*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    int rInpCols;
//    double * inputDeltaElemR;
//    for(int r=0; r<input->rows; r++){
//        rInpCols = r*input->cols;
//        inputDeltaElemR = inputDelta->elem[r];
//        vector<bool> &inputActiveR = inputActivity->activeUnits[r];
//        for(int c=0; c<input->cols; c++)
//            if (inputActiveR[c] && outputActivity->activeUnits[rInpCols+c])
//                inputDeltaElemR[c]+=outputDelta->elem[rInpCols+c];
//    }
//}
//
//bool Flatten2D::HasWeightsDependency(){
//    return 0;
//}
//
//void Flatten2D::BackpropagateDroppedUnits(int from, int to){
//    activityMatrix* inputActivity  = static_cast<activityMatrix*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    int rows = inputActivity->activeUnits.size();
//    int cols = inputActivity->activeUnits[0].size();
//
//    for(int r=0; r<rows; r++)
//        for(int c=0; c<cols; c++)
//            if (!outputActivity->activeUnits[r*cols+c])
//                inputActivity->activeUnits[r][c]=0;
//}
//
//
//
//Flatten3D::Flatten3D(){
//}
//
//void Flatten3D::ForwardPass(int from, int to){
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    vect* output=static_cast<vect*>(layersData->layerList[to]);
//
//    activityTensor* inputActivity  = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityVect*   outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    matrix* inputElemD;
//    double * inputElemDR;
//    int dRowsCols;
//    int dRowsColsPlusRCols;
//    for(int d=0; d<input->depth; d++){
//        inputElemD = &(input->elem[d]);
//        dRowsCols=d*input->rows*input->cols;
//        vector<vector<bool> >& inputActD = inputActivity->activeUnits[d];
//
//        for(int r=0; r<input->rows; r++){
//            inputElemDR = inputElemD->elem[r];
//            dRowsColsPlusRCols = dRowsCols + r*input->cols;
//            vector<bool> & inputActDR = inputActD[r];
//
//            for(int c=0; c<input->cols; c++)
//                if (inputActDR[c] && outputActivity->activeUnits[dRowsColsPlusRCols+c])
//                    output->elem[dRowsColsPlusRCols+c]+=inputElemDR[c];
//        }
//
//    }
//
//}
//
//void Flatten3D::BackwardPass(int from, int to, bool computeDelta, int trueClass){
//    if (!computeDelta) return;
//
//    tensor* input=static_cast<tensor*>(layersData->layerList[from]);
//    tensor* inputDelta=static_cast<tensor*>(deltas->layerList[from]);
//    vect* outputDelta=static_cast<vect*>(deltas->layerList[to]);
//
//    activityTensor* inputActivity  = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityVect*   outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    matrix* inputDeltaD;
//    double *inputDeltaDR;
//    int dRowsCols;
//    int dRowsColsPlusRCols;
//    for(int d=0; d<input->depth; d++){
//        dRowsCols=d*input->rows*input->cols;
//        inputDeltaD = &(inputDelta->elem[d]);
//        vector<vector<bool> >& inputActD = inputActivity->activeUnits[d];
//
//        for(int r=0; r<input->rows; r++){
//            dRowsColsPlusRCols = dRowsCols + r*input->cols;
//            inputDeltaDR = inputDeltaD->elem[r];
//            vector<bool> & inputActDR = inputActD[r];
//
//            for(int c=0; c<input->cols; c++)
//                if (inputActDR[c] && outputActivity->activeUnits[dRowsColsPlusRCols+c])
//                    inputDeltaDR[c]+=outputDelta->elem[dRowsColsPlusRCols+c];
//        }
//
//    }
//}
//
//bool Flatten3D::HasWeightsDependency(){
//    return 0;
//}
//
//void Flatten3D::BackpropagateDroppedUnits(int from, int to){
//    activityTensor* inputActivity  = static_cast<activityTensor*>(layersActivity->layerList[from]);
//    activityVect* outputActivity = static_cast<activityVect*>(layersActivity->layerList[to]);
//
//    int depth= inputActivity->activeUnits.size();
//    int rows = inputActivity->activeUnits[0].size();
//    int cols = inputActivity->activeUnits[0][0].size();
//
//    for(int d=0; d<depth; d++)
//        for(int r=0; r<rows; r++)
//            for(int c=0; c<cols; c++)
//                if (!outputActivity->activeUnits[d*rows*cols+r*cols+c])
//                    inputActivity->activeUnits[d][r][c]=0;
//}
