#include "vect.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "matrix.h"
#include "mathFunc.h"
#include <math.h>
#include "activityData.h"
using namespace std;
vect::vect(){
}

vect::vect(int len_): orderedData(len_){
}

float& vect::operator[] (int index)
{
    //cout<<"Operator []"<<endl;
    cout<<"mutator"<<endl;
    return elem[index];
}

const float& vect::operator[] (int index) const
{
    cout<<"accessor"<<endl;
    return elem[index];
}


void vect::Add(vect* addon, vector<int> &indexOutput){
    for(unsigned int i=0; i<indexOutput.size(); i++)
            elem[indexOutput[i] ]+=addon->elem[indexOutput[i] ];
}


//
//void vect::AddTrMatrVectProduct(matrix* A, vect* B, vector<int> &toBeComputed, vector<int> &bIndices){
//    float bElemI;
//    float *AelemI;
//    int toBeComputedSize = toBeComputed.size();
//    int bIndicesSize = bIndices.size();
//
//    for(int i=0; i<bIndicesSize; ++i)
//    {
//        bElemI=B->elem[bIndices[i] ];
//        AelemI = A->elem[bIndices[i] ];
//        for(int j=0; j<toBeComputedSize; ++j)
//            elem[toBeComputed[j] ]+=AelemI[toBeComputed[j] ] * bElemI;
//    }
//}

int vect::Dimensionality(){
    return 1;
}

void vect::StaticCastToVect(orderedData* arg){
    len = arg->len;
    elem = arg->elem;
}


void vect::SetToMatrixRow(matrix* M, int r){
    elem = M->Row(r);
    len = M->cols;
}

float* vect::elemLink(int j){
    return elem + j;
}

void vect::SubVect(vect* V, int len_){
    elem = V->elem;
    len = len_;
}

void vect::SubVect(vect* V, int startInd, int len_){
    elem = V->elemLink(startInd);
    len = len_;
}
