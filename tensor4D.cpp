#include "tensor4D.h"
#include <fstream>
#include "tensor.h"
#include <iostream>
#include "matrix.h"
#include "mathFunc.h"
using namespace std;
tensor4D::tensor4D(): number(0), depth(0), rows(0), cols(0){
    elem=NULL;
}

tensor4D::tensor4D(int number_, int depth_, int rows_, int cols_):
    orderedData(number_*depth_*rows_*cols_), number(number_), depth(depth_), rows(rows_), cols(cols_){
}

float * tensor4D::TLayer(int n){
    return elem+n*depth*rows*cols;
}

float& tensor4D::At(int n, int d, int r, int c){
    return elem[((n*depth+d)*rows+r)*cols+c];
}

int tensor4D::Ind(int n, int d, int r, int c){
    return ((n*depth+d)*rows+r)*cols+c;
}



void tensor4D::Print(){

tensor* t4_n = new tensor();
t4_n->SetSize(depth, rows, cols);
for(int n=0; n<number; n++){
        t4_n->PointToTensor(TLayer(n));
        cout<<"tensor4D; layer: "<<n<<": "<<endl;
        t4_n->Print();
        cout<<endl;
    }
}

void tensor4D::Reverse(tensor4D* T4D){
    tensor* this_n = new tensor();
    tensor* T4D_n = new tensor();
    for(int n=0; n<number; ++n){
        this_n->SetToTLayer(this, n);
        T4D_n->SetToTLayer(T4D, n);
        this_n->Reverse(T4D_n);
    }
    DeleteOnlyShell(this_n);
    DeleteOnlyShell(T4D_n);
}


int tensor4D::Dimensionality(){
    return 4;
}


void tensor4D::Sub4DTensor(tensor4D* T4D, int number_){
    elem = T4D->elem;
    number = number_;
    depth = T4D->depth;
    rows = T4D->rows;
    cols = T4D->cols;
    len = number * depth * rows * cols;
}


void tensor4D::Sub4DTensor(tensor4D* T4D, int startingNumber, int number_){
    elem = T4D->TLayer(startingNumber);
    number = number_;
    depth = T4D->depth;
    rows = T4D->rows;
    cols = T4D->cols;
    len = number * depth * rows * cols;
}
