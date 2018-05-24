#include "matrix.h"
#include "vect.h"
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include "mathFunc.h"
#include <math.h>
#include "bmp/EasyBMP.h"
#include "activityData.h"
#include "tensor.h"
using namespace std;

matrix::matrix(){
}

matrix::matrix(int rows_, int cols_): orderedData(rows_*cols_), rows(rows_), cols(cols_){
}

void matrix::PointToMatrix(int rows_, int cols_, float* elem_){
    rows=rows_;
    cols=cols_;
    len=rows_*cols_;
    elem=elem_;
}

void matrix::PointToMatrix(float *elem_){
    elem=elem_;
}

float& matrix::At(int r, int c){
    return elem[r*cols+c];
}

int matrix::Ind(int r, int c){
    return r*cols+c;
}

int matrix::IndRow(int r){
    return r*cols;
}

float* matrix::Row(int r){
    return elem+r*cols;
}

void matrix::SetSize(int rows_, int cols_){
    rows=rows_;
    cols=cols_;
    len=rows_*cols_;
}


void matrix::Print(){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            cout<<this->elem[i*cols+j]<<'\t';
        cout<<endl;
    }
}


void matrix::WriteToFile(char filename[]){
    ofstream f(filename);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            f<<elem[i*cols+j]<<'\t';
        f<<endl;
    }
    f.close();
}


void matrix::SaveAsImage(char filename[]){
    BMP Output;
    Output.SetSize(cols, rows);
    Output.SetBitDepth(24);
    for(int r=0; r<rows; r++)
        for(int c=0; c<cols; c++){
            Output(c,r)->Red   = round (255.0 * (At(r,c)+0.5) );
            Output(c,r)->Green = round (255.0*(At(r,c)+0.5));
            Output(c,r)->Blue  = round (255.0*(At(r,c)+0.5));
        }
    Output.WriteToFile(filename);
}

void matrix::SaveNormalizedAsImage(char filename[]){
    BMP Output;
    Output.SetSize(cols, rows);
    Output.SetBitDepth(24);
    float maxVal = this->Max();
    float minVal = this->Min();
    float fact;
    cout<<minVal<<" "<<maxVal<<endl;
    if (maxVal-minVal<1E-10) fact=0;
    else fact=255.0/(maxVal-minVal);
    cout<<fact<<endl;
    for(int r=0; r<rows; r++)
        for(int c=0; c<cols; c++){
            Output(c,r)->Red   = round  ((At(r,c)-minVal)*fact);
            Output(c,r)->Green = round  ((At(r,c)-minVal)*fact);
            Output(c,r)->Blue  = round  ((At(r,c)-minVal)*fact);
        }
    Output.WriteToFile(filename);
}

void matrix::BackwardFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* input, vect* biasGrad, vector<int>& indexOutput, vector<int>& indexInput){
    int outputDelta_size = indexOutput.size();
    int input_size = indexInput.size();
    float outputDelta_elemR;
    float * elem_r;
    int toComputeOutputDelta_r, toComputeInput_c;
    float *input_elem = input->elem;

    for(int r=0; r<outputDelta_size; ++r){
        toComputeOutputDelta_r = indexOutput[r];
        elem_r = this->Row(toComputeOutputDelta_r);
        outputDelta_elemR = outputDelta->elem[toComputeOutputDelta_r];
        biasGrad->elem[toComputeOutputDelta_r] += outputDelta_elemR;
        for(int c=0; c<input_size; ++c){
            toComputeInput_c = indexInput[c];
            elem_r[toComputeInput_c] += outputDelta_elemR * input_elem[toComputeInput_c];
        }
    }
}

void matrix::BackwardFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* input, vect* biasGrad, vector<int>& indexInput){
    int input_size = indexInput.size();
    float outputDelta_elemR;
    float * elem_r;
    int toComputeInput_c;
    float *input_elem = input->elem;

    for(int r=0; r<outputDelta->len; ++r){
        elem_r = this->Row(r);
        outputDelta_elemR = outputDelta->elem[r];
        biasGrad->elem[r] += outputDelta_elemR;
        for(int c=0; c<input_size; ++c){
            toComputeInput_c = indexInput[c];
            elem_r[toComputeInput_c] += outputDelta_elemR * input_elem[toComputeInput_c];
        }
    }
}


void matrix::BackwardFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* input, vect* biasGrad){
    float outputDelta_r;
    float * elem_r;
    int toComputeInput_c;
    float *input_elem = input->elem;
    int out_len = outputDelta->len;
    int input_len = input->len;
    for(int r=0; r<out_len; ++r){
        elem_r = this->Row(r);
        outputDelta_r = outputDelta->elem[r];
        biasGrad->elem[r] += outputDelta_r;
        for(int c=0; c<input_len; ++c){
            elem_r[c] += outputDelta_r * input_elem[c];
        }
    }
}


void matrix::BackwardFullyConnectedNoBiasOnlyGrad(orderedData* outputDelta, orderedData* input){
    float outputDelta_r;
    float * elem_r;
    int toComputeInput_c;
    float *input_elem = input->elem;
    int out_len = outputDelta->len;
    int input_len = input->len;
    for(int r=0; r<out_len; ++r){
        elem_r = this->Row(r);
        outputDelta_r = outputDelta->elem[r];
        for(int c=0; c<input_len; ++c){
            elem_r[c] += outputDelta_r * input_elem[c];
        }
    }
}




void matrix::BackwardCompressedInputFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* compressedInput, vect* biasGrad, vector<int>& indexInput){
    int input_size = indexInput.size();
    float outputDelta_elemR;
    float * elem_r;
    float *comprInput_elem = compressedInput->elem;

    for(int r=0; r<outputDelta->len; ++r){
        elem_r = this->Row(r);
        outputDelta_elemR = outputDelta->elem[r];
        biasGrad->elem[r] += outputDelta_elemR;
        for(int c=0; c<input_size; ++c){
            elem_r[indexInput[c] ] += outputDelta_elemR * comprInput_elem[c];
        }
    }
}

//void matrix::AddMatrMatrProduct(matrix* A, matrix* B){
//    cout<<"Call of non-optimized function"<<endl;
//    for(int i=0; i<A->rows; i++)
//        for(int j=0; j<B->cols; j++)
//            for(int k=0; k<A->cols; k++)
//                elem[i][j]+=A->elem[i][k]*B->elem[k][j];
//}


int matrix::Dimensionality(){
    return 2;
}


void matrix::SetToTensorLayer(tensor* T, int d){
    elem = T->Layer(d);
    rows = T->rows;
    cols = T->cols;
    len  = rows*cols;
}

void matrix::Reverse(matrix* M){
    for(int j=0; j<len; ++j)
        elem[j] = M->elem[len - 1 - j];
}


void matrix::SubMatrix(matrix* M, int rows_){
    elem = M->elem;
    rows = rows_;
    cols = M->cols;
    len = rows * cols;
}

void matrix::SubMatrix(matrix* M, int startRow, int rows_){
    elem = M->Row(startRow);
    rows = rows_;
    cols = M->cols;
    len = rows * cols;
}

void matrix::EigenDecompose(vect* eigenValues, matrix* eigenVectors){
    if (rows == 2 && cols == 2){
        float T = elem[0] + elem[3];
        float D = elem[0] * elem[3] - sqr(elem[1]);
        float delta = sqrt(sqr(T)/4.0 - D);
        eigenValues->elem[0] = T/2.0 - delta;
        eigenValues->elem[1] = T/2.0 + delta;

        float norm1 = sqrt(sqr(eigenValues->elem[0] - elem[3]) + sqr(elem[1]) );
        float norm2 = sqrt(sqr(eigenValues->elem[1] - elem[3]) + sqr(elem[1]) );

        eigenVectors->elem[0] = (eigenValues->elem[0] - elem[3]) / norm1;
        eigenVectors->elem[1] = (eigenValues->elem[1] - elem[3]) / norm2;
        eigenVectors->elem[2] = elem[1] / norm1;
        eigenVectors->elem[3] = elem[1] / norm2;
    }

    else{
        cout<<"Error: for bigger dimensionality eigen decomposition is not implemented"<<endl;
    }
}

float matrix::Interpolate(float x, float y){
    int r0 = int(y * (rows - 1));
    int c0 = int(x * (cols - 1));

    float f_00 = this->At(r0, c0);
    float f_01 = this->At(r0+1, c0);
    float f_10 = this->At(r0, c0+1);
    float f_11 = this->At(r0+1, c0+1);

    float a_1 = f_00;
    float a_x = f_10 - a_1;
    float a_y = f_01 - a_1;
    float a_xy = f_11 - a_x - a_y - a_1;

    float x_ = x * (cols - 1) - c0;
    float y_ = y * (rows - 1) - r0;

    return a_x * x_ + a_y * y_ + a_xy * x_ * y_ + a_1;
}

void matrix::CopySubMatrixMultiplied(float lamb, matrix* M, int border){
    float * this_r, * M_r;
    for(int r = border; r<M->rows - border; ++r){
        this_r = this->Row(r);
        M_r = M->Row(r);
        for(int c=border; c<cols - border; ++c)
            this_r[c] = lamb * M_r[c];
    }
}

void matrix::AddSubMatrix(float lamb, matrix* M, int border){
    float * this_r, * M_r;
    for(int r = border; r<M->rows - border; ++r){
        this_r = this->Row(r);
        M_r = M->Row(r);
        for(int c=border; c<cols - border; ++c)
            this_r[c] += lamb * M_r[c];
    }
}
