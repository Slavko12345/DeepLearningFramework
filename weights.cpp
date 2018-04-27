#include "weights.h"
#include "tensor.h"
#include "vect.h"
#include "tensor4D.h"
#include "matrix.h"
#include <iostream>
#include "mathFunc.h"
#include <math.h>
using namespace std;


void weights::SetModel()
{
    Nweights=4;
    weightList = new layerWeight[Nweights];

    weightList[0].SetStairsFullConvolution(3, 4, 4);

    weightList[1].SetStairsFullConvolution(35, 4, 4);

    weightList[2].SetStairsFullConvolution(67, 4, 4);

    weightList[3].SetFC(99, 10);
}

void layerWeight::SetStairsVert(int startDepth, int numStairs, int numStairConvolutions){
    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + 2 * numStairConvolutions * stair) * numStairConvolutions;
    }
    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetStairsHor(int numStairs, int numStairConvolutions){
    dataWeight = new tensor(numStairs * numStairConvolutions, 3, 2);
    bias = new vect(0);
}

void layerWeight::SetStairsSymmHor(int numStairs, int numStairConvolutions, int symmetryLevel){
    if (symmetryLevel == 0)
        dataWeight = new tensor(numStairs * numStairConvolutions, 3, 3);
    if (symmetryLevel == 1)
        dataWeight = new tensor(numStairs * numStairConvolutions, 3, 2);
    if (symmetryLevel == 2)
        dataWeight = new tensor(numStairs * numStairConvolutions, 2, 2);
    if (symmetryLevel == 3)
        dataWeight = new tensor(numStairs * numStairConvolutions, 1, 3);
    if (symmetryLevel == 4)
        dataWeight = new tensor(numStairs * numStairConvolutions, 1, 2);
    bias = new vect(0);
}

void layerWeight::SetSymmetricConvolutionRelu(int startDepth, int numStairs, int numStairConvolutions, int symmetryLevel){
    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + numStairConvolutions * stair) * numStairConvolutions;
    }

    if (symmetryLevel == 0)
        dataWeight = new tensor(vertLen, 3, 3);
    if (symmetryLevel == 1)
        dataWeight = new tensor(vertLen, 3, 2);
    if (symmetryLevel == 2)
        dataWeight = new tensor(vertLen, 2, 2);
    if (symmetryLevel == 3)
        dataWeight = new tensor(vertLen, 1, 3);
    if (symmetryLevel == 4)
        dataWeight = new tensor(vertLen, 1, 2);
    bias = new vect(0);
}


void layerWeight::SetStairsFullConvolution(int startDepth, int numStairs, int numStairConvolutions, int symmetryLevel, bool biasIncluded){
    int vertLen = 0;
    for(int stair=0; stair<numStairs; ++stair){
        vertLen += (startDepth + 2 * numStairConvolutions * stair) * numStairConvolutions;
    }

    if (symmetryLevel == 0)
        dataWeight = new tensor(vertLen, 3, 3);
    if (symmetryLevel == 1)
        dataWeight = new tensor(vertLen, 3, 2);
    if (symmetryLevel == 2)
        dataWeight = new tensor(vertLen, 2, 2);
    if (symmetryLevel == 3)
        dataWeight = new tensor(vertLen, 1, 3);
    if (symmetryLevel == 4)
        dataWeight = new tensor(vertLen, 1, 2);

    if (biasIncluded)
        bias = new vect(numStairs * numStairConvolutions);
    else
        bias = new vect(0);
}


void layerWeight::SetSequentialVert(int startDepth, int numStairs, int numStairConvolutions){
    int vertLen = startDepth * numStairConvolutions + 2 * sqr(numStairConvolutions) * (numStairs - 1);
    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetSequentialHor(int numStairs, int numStairConvolutions, int symmetryLevel){
    if (symmetryLevel == 0)
        dataWeight = new tensor(numStairs * numStairConvolutions, 3, 3);
    if (symmetryLevel == 1)
        dataWeight = new tensor(numStairs * numStairConvolutions, 3, 2);
    if (symmetryLevel == 2)
        dataWeight = new tensor(numStairs * numStairConvolutions, 2, 2);
    if (symmetryLevel == 3)
        dataWeight = new tensor(numStairs * numStairConvolutions, 1, 3);
    if (symmetryLevel == 4)
        dataWeight = new tensor(numStairs * numStairConvolutions, 1, 2);
    bias = new vect(0);
}



void layerWeight::SetStandard(int initialNumber, int nConvolutions){
    dataWeight = new tensor(nConvolutions * (nConvolutions + initialNumber - 1), 3, 3);
    bias = new vect(nConvolutions);
}

void layerWeight::SetStandardLinear(int startNumber, int nConvolutions){
    dataWeight = new vect(nConvolutions * (nConvolutions + startNumber - 1));
    bias = new vect(nConvolutions);
}

void layerWeight::SetFC(int fromSize, int toSize){
    dataWeight = new matrix(toSize, fromSize);
    bias = new vect(toSize);
}

void layerWeight::SetFCNoBias(int fromSize, int toSize){
    dataWeight = new matrix(toSize, fromSize);
    bias = new vect(0);
}


void layerWeight::SetStandardMultiple(int startDepth, int nConvolutions){
    int depth_weight = startDepth * (power(3, nConvolutions) - 1) / 2;
    dataWeight = new tensor(depth_weight, 3, 3);
    bias = new vect(depth_weight);
}


void layerWeight::SetBottleneckStandardVert(int initialNumber, int nConvolutions){
    dataWeight = new tensor(nConvolutions * (nConvolutions + initialNumber - 1), 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 3, 3);
    bias = new vect(nConvolutions);
}

void layerWeight::SetBottleneckStandardLimitedVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth){
    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startNumber - alwaysPresentDepth + 2 * j, limitDepth);

    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardLimitedHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 3, 3);
    bias = new vect(nConvolutions);
}

void layerWeight::SetBottleneckStandardRandomSymmetricVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth){
    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startNumber - alwaysPresentDepth + 2 * j, limitDepth);

    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardRandomSymmetricHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 3, 2);
    bias = new vect(nConvolutions);
}


void layerWeight::SetBottleneckStandardRandomSymmetricNoBiasVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth){
    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startNumber - alwaysPresentDepth + 2 * j, limitDepth);

    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardRandomSymmetricNoBiasHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 3, 2);
    bias = new vect(0);
}




void layerWeight::SetBottleneckStandardRandomFullySymmetricVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth){
    int vertLen = alwaysPresentDepth * nConvolutions;
    for(int j=0; j<nConvolutions; ++j)
        vertLen += min(startNumber - alwaysPresentDepth + 2 * j, limitDepth);

    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardRandomFullySymmetricHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 1, 2);
    bias = new vect(nConvolutions);
}





void layerWeight::SetBottleneckStandardRandomLimitedVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth){
    int vertLen = (alwaysPresentDepth + limitDepth) * nConvolutions;

    dataWeight = new tensor(vertLen, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardRandomLimitedHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 3, 3);
    bias = new vect(nConvolutions);
}


void layerWeight::SetBottleneckStandardReluVert(int initialNumber, int nConvolutions){
    dataWeight = new tensor(nConvolutions * (nConvolutions + 2 * initialNumber - 1) / 2, 1, 1);
    bias = new vect(0);
}

void layerWeight::SetBottleneckStandardReluHor(int nConvolutions){
    dataWeight = new tensor(nConvolutions, 3, 3);
    bias = new vect(nConvolutions);
}

void weights::SetToZero(){
    for(int w=0; w<Nweights; w++){
        weightList[w].dataWeight->SetToZero();
        weightList[w].bias->SetToZero();
    }
}

double weights::MaxAbs(){
    double maxAbs = -1;
    double weightsMax, biasMax;
    for(int w=0; w<Nweights; w++){
        weightsMax = weightList[w].dataWeight->MaxAbs();
        biasMax = weightList[w].bias->MaxAbs();
        maxAbs = max(maxAbs, weightsMax);
        maxAbs = max(maxAbs, biasMax);
    }
    return maxAbs;
}

void weights::SetToRandomValues(double maxAbs){
    for(int w=0; w<Nweights; w++){
        weightList[w].dataWeight->SetToRandomValues(maxAbs);
        weightList[w].bias->SetToRandomValues(maxAbs);
    }
}

void weights::Add(weights* addon){
    layerWeight* addon_weightList = addon->weightList;
    for(int w=0; w<Nweights; ++w){
        weightList[w].dataWeight->Add(addon_weightList[w].dataWeight);
        weightList[w].bias->Add(addon_weightList[w].bias);
    }
}

void weights::Add(double lamb, weights* addon){
    for(int w=0; w<Nweights; w++){
        weightList[w].dataWeight->Add(lamb, addon->weightList[w].dataWeight);
        weightList[w].bias->Add(lamb, addon->weightList[w].bias);
    }
}

void weights::SetToLinearCombination(double a1, double a2, weights* W1, weights* W2){
    for(int w=0; w<Nweights; ++w){
        weightList[w].dataWeight->SetToLinearCombination(a1, a2, W1->weightList[w].dataWeight,  W2->weightList[w].dataWeight);
        weightList[w].bias      ->SetToLinearCombination(a1, a2, W1->weightList[w].bias,        W2->weightList[w].bias);
    }
}

void weights::Copy(weights* W){
    for(int w=0; w<Nweights; w++){
        weightList[w].dataWeight->Copy(W->weightList[w].dataWeight);
        weightList[w].bias->Copy(W->weightList[w].bias);
    }
}

void weights::CopyMultiplied(double lamb, weights* W){
    for(int w=0; w<Nweights; w++){
        weightList[w].dataWeight->CopyMultiplied(lamb, W->weightList[w].dataWeight);
        weightList[w].bias->CopyMultiplied(lamb, W->weightList[w].bias);
    }
}

void weights::Print(){
    for(int j=0; j<Nweights; j++){
        weightList[j].dataWeight->Print();
        cout<<endl;
        weightList[j].bias->Print();
        cout<<endl;
    }
}

void weights::WriteToFile(char fileName[]){
    ofstream f(fileName);
    for(int j=0; j<Nweights; j++){
        weightList[j].dataWeight->WriteToFile(f);
        weightList[j].bias->WriteToFile(f);
    }
    f.close();
}

void weights::ReadFromFile(char filename[]){
    ifstream f(filename);
    for(int j=0; j<Nweights; j++){
        weightList[j].dataWeight->ReadFromFile(f);
        weightList[j].bias->ReadFromFile(f);
    }
    f.close();
}

void weights::RmspropUpdate(weights* grad, weights* MS, double k1, double k2, double Step){
    for(int j=0; j<Nweights; j++){
        weightList[j].dataWeight->RmspropUpdate(grad->weightList[j].dataWeight, MS->weightList[j].dataWeight, k1, k2, Step);
        weightList[j].bias->RmspropUpdate(grad->weightList[j].bias, MS->weightList[j].bias, k1, k2, Step);
    }
}

void weights::AdamUpdate(weights* grad, weights* Moment, weights* MS, double k1, double k2, double Step){
    for(int j=0; j<Nweights; ++j){
        weightList[j].dataWeight->AdamUpdate(grad->weightList[j].dataWeight, Moment->weightList[j].dataWeight, MS->weightList[j].dataWeight, k1, k2, Step);
        weightList[j].bias->AdamUpdate(grad->weightList[j].bias, Moment->weightList[j].bias, MS->weightList[j].bias, k1, k2, Step);
    }
}


weights::~weights(){
    for(int j=0; j<Nweights; ++j){
        delete weightList[j].dataWeight;
        delete weightList[j].bias;
    }
    delete [] weightList;
}

int weights::GetWeightLen(){
    int len = 0;
    for(int j=0; j<Nweights; ++j){
        len+=weightList[j].dataWeight->len;
        len+=weightList[j].bias->len;
    }
    return len;
}


void weights::Multiply(double lamb){
    for(int j=0; j<Nweights; ++j){
        weightList[j].dataWeight->Multiply(lamb);
        weightList[j].bias->Multiply(lamb);
    }
}



double InnerProduct(weights* w1, weights* w2){
    double res = 0;
    for(int j=0; j<w1->Nweights; ++j){
        res += InnerProduct(w1->weightList[j].dataWeight, w2->weightList[j].dataWeight);
        res += InnerProduct(w1->weightList[j].bias, w2->weightList[j].bias);
    }
    return res;
}


void FormOrthonormalBasis(weights* grad, weights* moment, weights* d1, weights* d2, double & g_g){
    g_g = InnerProduct(grad, grad);
    double g_m = InnerProduct(grad, moment);
    double m_m = InnerProduct(moment, moment);

    double det = m_m * g_g - sqr(g_m);
    double a1 = sqrt( g_g / det );
    double a2 = -g_m / sqrt(g_g * det);

    d1->CopyMultiplied(1.0/sqrt(g_g), grad);
    d2->SetToLinearCombination(a1, a2, moment, grad);
}


void FormHessianInSubspace(weights* d1, weights* d2, weights* Hd1, weights* Hd2, matrix* B){
    B->At(0,0) = InnerProduct(d1, Hd1);
    B->At(0,1) = InnerProduct(d1, Hd2);
    B->At(1,0) = InnerProduct(d2, Hd1);
    B->At(1,1) = InnerProduct(d2, Hd2);

    B->At(0,1) = ( B->At(0,1) + B->At(1,0) ) / 2.0;
    B->At(1,0) = B->At(0,1);
}
