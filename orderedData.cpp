#include "orderedData.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "mathFunc.h"
#include "activityData.h"
#include <math.h>
#include "randomGenerator.h"
#include "vect.h"
#include "matrix.h"
#include <algorithm>
using namespace std;

orderedData::orderedData(){
    elem=NULL;
    len=0;
}

orderedData::orderedData(int len_){
    AllocateMemory(len_);
}

orderedData::~orderedData(){
    delete[] elem;
}

void orderedData::AllocateMemory(int len_){
    len = len_;
    elem = new double[len];
}

void orderedData::SetToZero(){
    for(int i=0; i<len; ++i)
        elem[i]=0;
}

void orderedData::SetToZeroStartingFrom(int startingIndex){
    for(int i=startingIndex; i<len; ++i)
        elem[i]=0;
}

void orderedData::SetToValue(double val){
    for(int i=0; i<len; ++i)
        elem[i]=val;
}

void orderedData::SetToRandomValues(double maxAbs){
    for(int j=0; j<len; ++j)
        elem[j]=randomGenerator::generateDouble(maxAbs);
}

void orderedData::Print(){
    for(int i=0; i<len; ++i)
        cout<<elem[i]<<'\t';
    cout<<endl;
}

void orderedData::ReadFromFile(char filename[]){
    ifstream f(filename);
    if (!f) cout<<"File not found"<<endl;
    for(int j=0; j<len; ++j)
        f>>elem[j];
    f.close();
}

void orderedData::ReadFromFile(ifstream &f){
    for(int j=0; j<len; ++j)
        f>>elem[j];
}

void orderedData::WriteToFile(char filename[]){
    ofstream f(filename);
    for(int j=0; j<len; ++j)
        f<<elem[j];
    f.close();
}

void orderedData::WriteToFile(ofstream &f){
    for(int j=0; j<len; ++j)
        f<<elem[j];
}

void orderedData::Copy(orderedData* A){
    double * A_elem = A->elem;
    for(int i=0; i<len; ++i)
        elem[i]=A_elem[i];
}

void orderedData::CopyToSubpart(orderedData* A){
    double * A_elem = A->elem;
    for(int i = 0; i < A->len; ++i)
        elem[i] = A_elem[i];
}

void orderedData::CopyMultiplied(double lamb, orderedData* A){
    double * A_elem = A->elem;
    for(int i=0; i<len; ++i)
        elem[i] = lamb * A_elem[i];
}

void orderedData::MatrProd(matrix* M, vect* v){
    double * M_r, * v_elem = v->elem;
    double temp;
    for(int r=0; r<M->rows; ++r){
        M_r = M->Row(r);
        temp=0;
        for(int c=0; c<M->cols; ++c)
            temp += M_r[c] * v_elem[c];
        elem[r] = temp;
    }
}

void orderedData::TrMatrProd(matrix* M, vect* v){
    this->SetToZero();
    double* M_r;
    double v_r;
    for(int r=0; r<M->rows; ++r){
        M_r = M->Row(r);
        v_r = v->elem[r];
        for(int c=0; c<M->cols; ++c)
            elem[c] += M_r[c] * v_r;
    }
}




void orderedData::Add(orderedData* addon){
    double *addon_elem = addon->elem;
    for(int i=0; i<len; ++i)
        elem[i]+=addon_elem[i];
}

void orderedData::Add(double lamb, orderedData* addon){
    double *addon_elem = addon->elem;
    for(int i=0; i<len; ++i)
        elem[i]+=lamb * addon_elem[i];
}

void orderedData::Add(double addon){
    for(int i=0; i<len; ++i)
        elem[i]+=addon;
}


void orderedData::SetToLinearCombination(double a1, double a2, orderedData* A1, orderedData* A2){
    double* A1_elem = A1->elem, *A2_elem = A2->elem;
    for(int j=0; j<A1->len; ++j)
        elem[j] = a1 * A1_elem[j] + a2 * A2_elem[j];
}


void orderedData::AddThisStartingFrom(int this_startingIndex, orderedData* addon){
    double *addon_elem = addon->elem;
    double *elem_start = this->elem + this_startingIndex;
    for(int i=0; i<addon->len; ++i)
        elem_start[i]+=addon_elem[i];
}


void orderedData::AddThisStartingFromOnlyActive(int this_startingIndex, orderedData* addon, activityData* this_activity){
    double *addon_elem = addon->elem;
    double *elem_start = this->elem + this_startingIndex;
    bool* this_activity_start = this_activity->activeUnits + this_startingIndex;
    for(int i=0; i<addon->len; ++i)
        elem_start[i] += this_activity_start[i] * addon_elem[i];
}


void orderedData::AddAddonStartingFrom(int addonStartingIndex, orderedData* addon){
    double *addon_elem_start = addon->elem + addonStartingIndex;
    for(int i=0; i<len; ++i)
        elem[i]+=addon_elem_start[i];
}


void orderedData::AddAddonStartingFromOnlyActive(int addonStartingIndex, orderedData* addon, activityData* this_activity){
    double *addon_elem_start = addon->elem + addonStartingIndex;
    bool* this_activity_ = this_activity->activeUnits;
    for(int i=0; i<len; ++i)
        elem[i] += this_activity_[i] * addon_elem_start[i];
}




void orderedData::AddPointwiseFunction(orderedData* inp, mathFunc* func){
    double *inp_elem = inp->elem;
    for(int j=0; j<len; ++j)
        elem[j]+=func->f(inp_elem[j]);
}

void orderedData::DroppedAddPointwiseFunction(orderedData* inp, mathFunc* func, activityData* inputActivity, activityData* outputActivity){
    bool* activeOut = outputActivity->activeUnits;
    double* inp_elem = inp->elem;
    for(int j=0; j<len; j++)
        elem[j] += activeOut[j] * func->f(inp_elem[j]);
//        if (inputActivity->activeUnits[j] && outputActivity->activeUnits[j])
//            elem[j]+=func->f(inp->elem[j]);
}


void orderedData::AddPointwiseFuncDerivMultiply(orderedData* inp, orderedData* funcArg, mathFunc* func){
    double* funcArg_elem = funcArg->elem;
    double * inp_elem = inp->elem;
    for(int j=0; j<len; ++j)
        elem[j]+=inp_elem[j]*(func->df(funcArg_elem[j]));
}

void orderedData::DroppedAddPointwiseFuncDerivMultiply(orderedData* inp, orderedData* funcArg, mathFunc* func, activityData* inputActivity, activityData* outputActivity){
    double* funcArg_elem = funcArg->elem;
    double * inp_elem = inp->elem;
    bool* activeOut = outputActivity->activeUnits;

    for(int j=0; j<len; ++j)
        elem[j] += activeOut[j] * inp_elem[j] * func->df(funcArg_elem[j]);
//        if (inputActivity->activeUnits[j] && outputActivity->activeUnits[j])
//            elem[j]+=inp->elem[j]*(func->df(funcArg->elem[j]));
}


void orderedData::Multiply(double lamb){
    for(int i=0; i<len; ++i)
        elem[i]*=lamb;
}

void orderedData::PointwiseMultiply(orderedData* A){
    double *A_elem = A->elem;
    for(int i=0; i<len; ++i)
        elem[i] *= A_elem[i];
}

double orderedData::Sum(){
    double res=0;
    for(int i=0; i<len; ++i)
        res+=elem[i];
    return res;
}

double orderedData::SqNorm(){
    double res=0;
    for(int i=0; i<len; ++i)
        res+=sqr(elem[i]);
    return res;
}

double orderedData::Max(){
    double res=elem[0];
    for(int j=1; j<len; j++)
        if (res<elem[j]) res=elem[j];
    return res;
}

double orderedData::Min(){
    double res=elem[0];
    for(int j=1; j<len; j++)
        if (res>elem[j]) res=elem[j];
    return res;
}

int orderedData::ArgMax(){
    double res=elem[0], ind=0;
    for(int j=1; j<len; j++)
        if (res<elem[j]){
            res=elem[j];
            ind=j;
        }
    return ind;
}

double orderedData::MaxAbs(){
    double maxAbs = -1;
    for(int j=0; j<len; ++j)
        maxAbs = max(maxAbs, fabs(elem[j]) );
    return maxAbs;
}

void orderedData::RmspropUpdate(orderedData* grad, orderedData* MS, double k1, double k2, double Step){
    double *MS_elem = MS->elem, *grad_elem = grad->elem;
    for(int j=0; j<len; j++){
        MS_elem[j] = k1*MS_elem[j] + k2*sqr(grad_elem[j]);
        elem[j] -= Step*grad_elem[j] / max(sqrt(MS_elem[j]), 0.0000001);
    }
}

void orderedData::AdamUpdate(orderedData* grad, orderedData* Moment, orderedData* MS, double k1, double k2, double Step){
    double *MS_elem = MS->elem, *grad_elem = grad->elem, *Moment_elem = Moment->elem;


    for(int j=0; j<len; ++j)
        Moment_elem[j] = k1*Moment_elem[j]+k2*grad_elem[j];

    for(int j=0; j<len; ++j)
        MS_elem[j]=k1*MS_elem[j]+k2*sqr(grad_elem[j]);

    for(int j=0; j<len; ++j)
        elem[j]-=Step*Moment_elem[j]/(sqrt(MS_elem[j])+0.0000001);

//    for(int j=0; j<len; ++j){
//        Moment_elem[j] = k1*Moment_elem[j]+k2*grad_elem[j];
//        MS_elem[j]=k1*MS_elem[j]+k2*sqr(grad_elem[j]);
//        elem[j]-=Step*Moment_elem[j]/(sqrt(MS_elem[j])+0.0000001);
//    }
}

void orderedData::SetDroppedElementsToZero(activityData* mask){
    if (!mask->dropping) return;
    bool* mask_active = mask->activeUnits;
    for(int j=0; j<len; ++j){
        elem[j] *= mask_active[j];
    }
}

void orderedData::SetDroppedElementsToZero(activityData* mask, int maxLen){
    if (!mask->dropping) return;
    bool* mask_active = mask->activeUnits;
    for(int j=0; j<maxLen; ++j){
        elem[j] *= mask_active[j];
    }
}

void orderedData::SetDroppedElementsToZero(activityData* mask, int startingInd, int maxLen){
    if (!mask->dropping) return;
    bool* mask_active_start = mask->activeUnits + startingInd;
    double* elem_start = elem + startingInd;
    for(int j=0; j<maxLen; ++j)
        elem_start[j] *= mask_active_start[j];
}

void orderedData::SetToReluFunction(){
    for(int j=0; j<len; ++j)
        elem[j] *= (elem[j]>0);
}

void orderedData::BackwardRelu(orderedData* input){
    double *inp_elem = input->elem;
    for(int j=0; j<len; ++j)
        elem[j] *= (inp_elem[j] > 0);
}

void orderedData::SetToMinReluFunction(orderedData* inp){
    double *inp_elem = inp->elem;
    for(int j=0; j<len; ++j)
        elem[j] = inp_elem[j] * (inp_elem[j]<0);
}

void orderedData::BranchMaxMin(orderedData* inp){
    double *inp_elem = inp->elem;
    for(int j=0; j<len; ++j){
        elem[j] = inp_elem[j] * (inp_elem[j]<0);
        inp_elem[j] -= elem[j];
    }
}


void orderedData::ListNonzeroElements(vector<int> & index){
    index.resize(0);
    for(int j=0; j<len; ++j)
        if (fabs(elem[j])>1E-10)
            index.push_back(j);
}

void orderedData::ListNonzeroElements(vector<int> & index, orderedData* compressed){
    index.resize(0);
    double* compressed_elem = compressed->elem;
    for(int j=0; j<len; ++j)
        if (fabs(elem[j])>1E-10){
            compressed_elem[index.size()] = elem[j];
            index.push_back(j);
        }
}



void orderedData::ListNonzeroActiveElements(vector<int> & index, activityData* activity){
    if (!activity->dropping){
        this->ListNonzeroElements(index);
        return;
    }

    bool* activity_ = activity->activeUnits;
    index.resize(0);
    for(int j=0; j<len; j++)
        if (activity_[j] && fabs(elem[j])>1E-10)
            index.push_back(j);
}

void orderedData::ListNonzeroActiveElements(vector<int> & index, activityData* activity, orderedData* compressed){
    if (!activity->dropping){
        this->ListNonzeroElements(index, compressed);
        return;
    }

    bool* activity_ = activity->activeUnits;
    double* compressed_elem = compressed->elem;
    index.resize(0);
    for(int j=0; j<len; j++)
        if (activity_[j] && fabs(elem[j])>1E-10){
            compressed_elem[index.size()] = elem[j];
            index.push_back(j);
        }
}




void orderedData::ListActiveElements(vector<int> & index, activityData* activity){
    if (!activity->dropping){
        this->ListAll(index);
        return;
    }

    bool* activity_ = activity->activeUnits;
    index.resize(0);
    for(int j=0; j<len; j++)
        if (activity_[j])
            index.push_back(j);
}

void orderedData::ListActiveElements(vector<int> & index, activityData* activity, orderedData* compressed){
    if (!activity->dropping){
        this->ListAll(index, compressed);
        return;
    }

    bool* activity_ = activity->activeUnits;
    double* compressed_elem = compressed->elem;
    index.resize(0);
    for(int j=0; j<len; j++)
        if (activity_[j]){
            compressed_elem[index.size()] = elem[j];
            index.push_back(j);
        }
}



void orderedData::ListAll(vector<int> & index){
    index.resize(0);
    for(int j=0; j<len; ++j)
        index.push_back(j);
}

void orderedData::ListAll(vector<int> & index, orderedData* compressed){
    index.resize(0);
    double* compressed_elem = compressed->elem;

    for(int j=0; j<len; ++j){
        compressed_elem[index.size()] = elem[j];
        index.push_back(j);
    }
}



void orderedData::AddMatrVectProductBias(matrix* kernel, orderedData* input, vect* bias, vector<int> &indexInput, vector<int> &indexOutput){
    double temp;
    double* kernel_i;
    double *input_ = input->elem;
    double *bias_ = bias->elem;
    int indOutI, indInpJ;

    int indexInputSize  = indexInput.size();
    int indexOutputSize = indexOutput.size();

    for(int i=0; i<indexOutputSize; ++i){
        indOutI = indexOutput[i];
        temp=bias_[indOutI];
        kernel_i=kernel->Row(indOutI);

        for(int j=0; j<indexInputSize; ++j){
            indInpJ=indexInput[j];
            temp+=kernel_i[indInpJ] * input_[indInpJ];
        }
        elem[indOutI] += temp;
    }
}


void orderedData::AddMatrVectProductBias(matrix* kernel, orderedData* input, vect* bias, vector<int> &indexInput){
    double temp;
    double* kernel_i;
    double *input_ = input->elem;
    double *bias_ = bias->elem;
    int indInpJ;

    int indexInputSize  = indexInput.size();

    for(int indOutI = 0; indOutI < len; ++indOutI){
        temp = bias_[indOutI];
        kernel_i = kernel->Row(indOutI);

        for(int j=0; j<indexInputSize; ++j){
            indInpJ = indexInput[j];
            temp += kernel_i[indInpJ] * input_[indInpJ];
        }
        elem[indOutI] += temp;
    }
}

void orderedData::AddMatrVectProductBias(matrix* kernel, orderedData* input, vect* bias){
    double temp;
    double* kernel_i;
    double *input_ = input->elem;
    double *bias_ = bias->elem;
    const int input_len = input->len;

    //int indexInputSize  = indexInput.size();

    for(int i = 0; i < len; ++i){
        temp = bias_[i];
        kernel_i = kernel->Row(i);

        for(int j=0; j<input_len; ++j){
            temp += kernel_i[j] * input_[j];
        }
        elem[i] += temp;
    }
}

void orderedData::AddMatrVectProduct(matrix* kernel, orderedData* input){
    double temp;
    double* kernel_i;
    double *input_ = input->elem;
    const int input_len = input->len;

    for(int i = 0; i < len; ++i){
        temp = 0;
        kernel_i = kernel->Row(i);

        for(int j=0; j<input_len; ++j){
            temp += kernel_i[j] * input_[j];
        }
        elem[i] += temp;
    }
}

void orderedData::AddMatrCompressedVectProductBias(matrix* kernel, orderedData* compressedInput, vect* bias, vector<int> &indexInput){
    double temp;
    double* kernel_i;
    double *bias_ = bias->elem;
    double *compressedInput_elem = compressedInput->elem;

    int indexInputSize  = indexInput.size();

    for(int indOutI = 0; indOutI < len; ++indOutI){
        temp = bias_[indOutI];
        kernel_i = kernel->Row(indOutI);

        for(int j=0; j<indexInputSize; ++j){
            temp += kernel_i[indexInput[j] ] * compressedInput_elem[j];
        }
        elem[indOutI] += temp;
    }
}




void orderedData::BackwardFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad, vect* biasGrad,
                                  vector<int> &indexInput, vector<int> &indexOutput){
    double *kernel_i;
    double *kernelGrad_i;
    double *inpDelta = elem;
    double *inp = input->elem;
    double *biasGrad_ = biasGrad->elem;

    double outputDelta_i;
    int indexInputSize  = indexInput.size();
    int indexOutputSize = indexOutput.size();
    int indOut_i, indInp_j;

    for(int i=0; i<indexOutputSize; ++i)
    {
        indOut_i = indexOutput[i];
        outputDelta_i=outputDelta->elem[indOut_i];
        kernel_i =         kernel->Row(indOut_i);
        kernelGrad_i = kernelGrad->Row(indOut_i);
        biasGrad_[indOut_i] += outputDelta_i;
        for(int j=0; j<indexInputSize; ++j){
            indInp_j = indexInput[j];
            inpDelta[indInp_j]+=kernel_i[indInp_j] * outputDelta_i;
            kernelGrad_i[indInp_j] += outputDelta_i * inp[indInp_j];
        }
    }
}


void orderedData::BackwardFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad, vect* biasGrad, vector<int> &indexInput){
    double *kernel_i;
    double *kernelGrad_i;
    double *inpDelta = elem;
    double *inp = input->elem;
    double *biasGrad_ = biasGrad->elem;

    double outputDelta_i;
    int indexInputSize  = indexInput.size();
    int indInp_j;

    for(int indOut_i=0; indOut_i<outputDelta->len; ++indOut_i)
    {
        outputDelta_i=outputDelta->elem[indOut_i];
        kernel_i =         kernel->Row(indOut_i);
        kernelGrad_i = kernelGrad->Row(indOut_i);
        biasGrad_[indOut_i] += outputDelta_i;
        for(int j=0; j<indexInputSize; ++j){
            indInp_j = indexInput[j];
            inpDelta[indInp_j] += kernel_i[indInp_j] * outputDelta_i;
            kernelGrad_i[indInp_j] += outputDelta_i * inp[indInp_j];
        }
    }
}

void orderedData::BackwardFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad, vect* biasGrad){
    double *kernel_i;
    double *kernelGrad_i;
    double *inpDelta = elem;
    double *inp = input->elem;
    double *biasGrad_ = biasGrad->elem;

    double outputDelta_i;
    int indInp_j;
    int out_len = outputDelta->len;
    int inp_len = input->len;

    for(int i=0; i<out_len; ++i)
    {
        outputDelta_i=outputDelta->elem[i];
        kernel_i =         kernel->Row(i);
        kernelGrad_i = kernelGrad->Row(i);
        biasGrad_[i] += outputDelta_i;
        for(int j=0; j<inp_len; ++j){
            inpDelta[j] += kernel_i[j] * outputDelta_i;
            kernelGrad_i[j] += outputDelta_i * inp[j];
        }
    }
}


void orderedData::BackwardFullyConnectedNoBias(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad){
    double *kernel_i;
    double *kernelGrad_i;
    double *inpDelta = elem;
    double *inp = input->elem;

    double outputDelta_i;
    int indInp_j;
    int out_len = outputDelta->len;
    int inp_len = input->len;

    for(int i=0; i<out_len; ++i)
    {
        outputDelta_i=outputDelta->elem[i];
        kernel_i =         kernel->Row(i);
        kernelGrad_i = kernelGrad->Row(i);
        for(int j=0; j<inp_len; ++j){
            inpDelta[j] += kernel_i[j] * outputDelta_i;
            kernelGrad_i[j] += outputDelta_i * inp[j];
        }
    }
}


void orderedData::BackwardCompressedInputFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* compressedInput, matrix* kernelGrad, vect* biasGrad,
                                         vector<int> &indexInput){
    double *kernel_i;
    double *kernelGrad_i;
    double *inpDelta = elem;
    double *comprInp = compressedInput->elem;
    double *biasGrad_ = biasGrad->elem;

    double outputDelta_i;
    int indexInputSize  = indexInput.size();
    int indInp_j;

    for(int indOut_i=0; indOut_i<outputDelta->len; ++indOut_i)
    {
        outputDelta_i=outputDelta->elem[indOut_i];
        kernel_i =         kernel->Row(indOut_i);
        kernelGrad_i = kernelGrad->Row(indOut_i);
        biasGrad_[indOut_i] += outputDelta_i;
        for(int j=0; j<indexInputSize; ++j){
            indInp_j = indexInput[j];
            inpDelta[indInp_j] += kernel_i[indInp_j] * outputDelta_i;
            kernelGrad_i[indInp_j] += outputDelta_i * comprInp[j];
        }
    }
}




double InnerProduct(orderedData* inp1, orderedData* inp2){
    double res = 0;
    double *inp1_elem = inp1->elem;
    double* inp2_elem = inp2->elem;
    for(int j=0; j<inp1->len; ++j)
        res += inp1_elem[j] * inp2_elem[j];
    return res;
}


double InnerProductSubMatrices(matrix* M1, matrix* M2, int border){
    double res = 0;
    double * M1_r, * M2_r;
    for(int r=border; r<M1->rows - border; ++r){
        M1_r = M1->Row(r);
        M2_r = M2->Row(r);
        for(int c=border; c<M1->cols - border; ++c){
            res += M1_r[c] * M2_r[c];
        }
    }
    return res;
}


void orderedData::MaxMinBackward(orderedData* minOutputDelta, orderedData* output){
    int startingIndex = len;
    double *out_elem = output->elem;
    double* minOutDelta_elem = minOutputDelta->elem;
    double* out_elem_start = out_elem + startingIndex;

    for(int j=0; j<len; ++j){
        elem[j] *= (out_elem[j]>0);
        minOutDelta_elem[j] *= (out_elem_start[j]<0);
        elem[j] += minOutDelta_elem[j];
    }
}

void orderedData::FindTrustRegionMinima(matrix* B, vect* r, double eps){
    matrix* eigenVectors = new matrix(B->rows, B->cols);
    vect* eigenValues = new vect(r->len);
    vect* rV = new vect(r->len);

    B->EigenDecompose(eigenValues, eigenVectors);
    cout<<"Eigenvalues: "<<endl;
    eigenValues->Print();

    cout<<"Eigenvectors: "<<endl;
    eigenVectors->Print();

    rV->TrMatrProd(eigenVectors, r);

    cout<<"V^T r:"<<endl;
    rV->Print();

    this->FindTrustRegionMinima(eigenValues, eigenVectors, rV, eps);
}

void orderedData::FindTrustRegionMinima(vect* eigenValues, matrix* eigenVectors, vect* rV, double eps){
    vect* solutionBasis = new vect(eigenValues->len);
    if (eigenValues->elem[0]>0){
        double globalSolSqNorm = 0;
        for(int j=0; j<rV->len; ++j)
            solutionBasis->elem[j] = rV->elem[j] / eigenValues->elem[j];
        if (solutionBasis->SqNorm() < sqr(eps) ){
            this->MatrProd(eigenVectors, solutionBasis);
            return;
        }
    }

    double lamb = 0;
    if (eigenValues->elem[0]<0){
        double addon = 1;
        while(1){
            lamb = - eigenValues->elem[0] + addon;
            if (TrustRegionFunc(lamb, eigenValues, rV, eps)>0)
                break;
            addon /= 2.0;
        }
    }
    double f_val;
    while(1){
        f_val = TrustRegionFunc(lamb, eigenValues, rV, eps);
        if (fabs(f_val)<1E-10) break;
        lamb -= f_val / TrustRegionFuncDeriv(lamb, eigenValues, rV, eps);
        //cout<<lamb<<endl;
    }

    cout<<"Final lambda: "<<lamb<<endl;
    if (lamb<0) cout<<"Error in Newton"<<endl;

    for(int j=0; j<rV->len; ++j)
        solutionBasis->elem[j] = rV->elem[j] / (eigenValues->elem[j] + lamb);
    this->MatrProd(eigenVectors, solutionBasis);
}

void orderedData::CalculateMeanStdDev(double & mean, double & stDev){
    mean = 0;
    stDev = 0;
    for(int j=0; j<len; ++j){
        mean += elem[j];
        stDev += sqr(elem[j]);
    }
    mean /= len;
    stDev /= len;
    stDev -= sqr(mean);
    stDev = sqrt(stDev);
}

void orderedData::NormalizeMeanStDev(orderedData* input, double & mean, double & stDev){
    this->CalculateMeanStdDev(mean, stDev);
    if (fabs(stDev < 1E-10)) stDev+=0.001;
    double invStDev = 1.0 / stDev;
    for(int j=0; j<len; ++j)
        elem[j] = (input->elem[j] - mean) * invStDev;
}

void orderedData::computeMedian(double * median, int * index){
    std::vector<double> dataCopy;
    dataCopy.reserve(len);
    for(int j=0; j<len; ++j)
        dataCopy.push_back(elem[j]);
    std::vector<double>::iterator middle = dataCopy.begin() + (len-1)/2;
    std::nth_element(dataCopy.begin(), middle, dataCopy.end());
    median[0] = *middle;
    for(int j=0; j<len; ++j)
        if (fabs(elem[j] - median[0]) < 1E-8){
            index[0] = j;
            break;
        }
}



void orderedData::computeMedianNonzero(double * median, int * index){
    std::vector<double> dataCopy;
    dataCopy.reserve(len);
    for(int j=0; j<len; ++j)
        if (fabs(elem[j])>1E-8)
            dataCopy.push_back(elem[j]);
    if (dataCopy.size() == 0){
        median[0] = 0;
        index[0] = 0;
        return;
    }
    std::vector<double>::iterator middle = dataCopy.begin() + (dataCopy.size() - 1) / 2;
    std::nth_element(dataCopy.begin(), middle, dataCopy.end());
    median[0] = *middle;
    for(int j=0; j<len; ++j)
        if (fabs(elem[j] - median[0]) < 1E-8){
            index[0] = j;
            break;
        }
}


void orderedData::computeQuartiles(double * quartiles, int * index, int numQuartiles){
    vector<double> dataCopy;
    dataCopy.reserve(len);
    for(int j=0; j<len; ++j)
        dataCopy.push_back(elem[j]);
    vector<double>::iterator quartileIterator_j;
    for(int j=0; j<numQuartiles; ++j){
        quartileIterator_j = dataCopy.begin() + round((dataCopy.size() - 1) * double(j + 1) / (numQuartiles + 1));
        nth_element(dataCopy.begin(), quartileIterator_j, dataCopy.end());
        quartiles[j] = * quartileIterator_j;
        for(int k=0; k<len; ++k)
            if (fabs(elem[k] - quartiles[j]) < 1E-8){
                index[j] = k;
                break;
            }
    }
}

void orderedData::computeQuartilesNonzero(double * quartiles, int * index, int numQuartiles){
    vector<double> dataCopy;
    dataCopy.reserve(len);
    for(int j=0; j<len; ++j)
        if (fabs(elem[j]) > 1E-8)
            dataCopy.push_back(elem[j]);

    if (dataCopy.size() == 0){
        for(int j=0; j<numQuartiles; ++j){
            quartiles[j] = 0;
            index[j] = 0;
        }
        return;
    }

    vector<double>::iterator quartileIterator_j;
    for(int j=0; j<numQuartiles; ++j){
        quartileIterator_j = dataCopy.begin() + round((dataCopy.size() - 1) * double(j + 1) / (numQuartiles + 1));
        nth_element(dataCopy.begin(), quartileIterator_j, dataCopy.end());
        quartiles[j] = * quartileIterator_j;
        for(int k=0; k<len; ++k)
            if (fabs(elem[k] - quartiles[j]) < 1E-8){
                index[j] = k;
                break;
            }
    }
}

void orderedData::AverageWith(orderedData * input){
    for(int j=0; j<len; ++j){
        elem[j] += input->elem[j];
        elem[j] *= 0.5;
    }
}

void orderedData::BackwardAverageWith(orderedData* inputDelta){
    for(int j=0; j<len; ++j){
        elem[j] *= 0.5;
        inputDelta->elem[j] = elem[j];
    }
}