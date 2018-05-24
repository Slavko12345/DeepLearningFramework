#include "tensor.h"
#include <fstream>
#include "matrix.h"
#include <iostream>
#include <stdlib.h>
#include "activityData.h"
#include "mathFunc.h"
#include "bmp/EasyBMP.h"
#include "tensor4D.h"
#include "randomGenerator.h"
#include "globals.h"
using namespace std;

tensor::tensor(){
}

tensor::tensor(int depth_, int rows_, int cols_): orderedData(depth_*rows_*cols_),
depth(depth_), rows(rows_), cols(cols_){
}

void tensor::SetSize(int depth_, int rows_, int cols_){
    depth = depth_;
    rows = rows_;
    cols = cols_;
    len = depth_*rows_*cols_;
}

void tensor::PointToTensor(float * elem_){
    elem = elem_;
}

void tensor::Print()
{
    matrix* t_d = new matrix;
    t_d->SetSize(rows, cols);
    for(int d=0; d<depth; ++d){
        t_d->PointToMatrix(this->Layer(d));
        t_d->Print();
        cout<<endl;
    }
    DeleteOnlyShell(t_d);
}

float& tensor::At(int d, int r, int c){
    return elem[(d*rows+r)*cols+c];
}

int tensor::Ind(int d, int r, int c){
    return (d*rows+r)*cols+c;
}

int tensor::Ind(int d){
    return d*rows*cols;
}

float* tensor::Layer(int d){
    return elem+d*rows*cols;
}


void tensor::SaveAsImage(char filename[]){
    BMP Output;
    Output.SetSize(cols, rows);
    Output.SetBitDepth(24);
    for(int r=0; r<rows; r++)
        for(int c=0; c<cols; c++){
            Output(c,r)->Red   = round (255.0 * (At(0, r, c)) );
            Output(c,r)->Green = round (255.0 * (At(1, r, c)) );
            Output(c,r)->Blue  = round (255.0 * (At(2, r, c)) );
        }
    Output.WriteToFile(filename);
}

void tensor::TransformByPoints(tensor* inputImage, matrix* coordsPoints){
    float a_x, a_y, a_xy, a_1;
    float b_x, b_y, b_xy, b_1;

    a_1 =   coordsPoints->At(0, 0);       b_1 = coordsPoints->At(0, 1);
    a_x =   coordsPoints->At(1, 0) - a_1; b_x = coordsPoints->At(1, 1) - b_1;
    a_y =   coordsPoints->At(2, 0) - a_1; b_y = coordsPoints->At(2, 1) - b_1;
    a_xy =  coordsPoints->At(3, 0) - a_x - a_y - a_1;
    b_xy =  coordsPoints->At(3, 1) - b_x - b_y - b_1;

    float x_source, y_source, x_dest, y_dest;
    float step_x = 1.0 / (cols - 1.0), step_y = 1.0 / (rows - 1.0);
    matrix* input_ch = new matrix();

    for(int ch = 0; ch < depth; ++ch){
        input_ch->SetToTensorLayer(inputImage, ch);
        for(int r = 0; r < rows; ++r)
            for(int c = 0; c < cols; ++c){
                x_source = step_x * c;
                y_source = step_y * r;
                x_dest = a_x * x_source + a_y * y_source + a_xy * x_source * y_source + a_1;
                y_dest = b_x * x_source + b_y * y_source + b_xy * x_source * y_source + b_1;
                this->At(ch, r, c) = input_ch->Interpolate(x_dest, y_dest);
            }
    }

    DeleteOnlyShell(input_ch);
}

void tensor::RandomlyTranform(tensor* inputImage){
    matrix* coords = new matrix(4, 2);
    float c_x, c_y;
    bool corner;
    for(int j=0; j<4; ++j){
        c_x = randomGenerator::generatePositiveFloat(AUGMENTATION_SIZE);
        c_y = randomGenerator::generatePositiveFloat(AUGMENTATION_SIZE);
        corner = randomGenerator::generateBool(CORNER_PROBABILITY);
        c_x *= (1 - corner);
        c_y *= (1 - corner);

        if (j % 2 == 0)
            coords->At(j, 0) = c_x;
        else
            coords->At(j, 0) = 1.0 - c_x;

        if (j<2)
            coords->At(j, 1) = c_y;
        else
            coords->At(j, 1) = 1.0 - c_y;

    }

    this->TransformByPoints(inputImage, coords);
    delete coords;
}



int tensor::Dimensionality(){
    return 3;
}

void tensor::SetToTLayer(tensor4D* T4D, int n){
    elem  = T4D->TLayer(n);
    depth = T4D->depth;
    rows  = T4D->rows;
    cols  = T4D->cols;
    len   = depth*rows*cols;
}


void tensor::SubTensor(tensor* T, int depth_){
    elem = T->elem;
    depth = depth_;
    rows = T->rows;
    cols = T->cols;
    len = depth * rows * cols;
}

void tensor::SubTensor(tensor* T, int startDepth, int depth_){
    elem = T->Layer(startDepth);
    depth = depth_;
    rows = T->rows;
    cols = T->cols;
    len = depth * rows * cols;
}

void tensor::SubLastTensor(tensor* T, int lastLayers){
    elem = T->Layer(T->depth - lastLayers);
    depth = lastLayers;
    rows = T->rows;
    cols = T->cols;
    len = depth * rows * cols;
}

void tensor::Reverse(tensor* T){
    matrix* T_d = new matrix();
    matrix* this_d = new matrix();
    for(int d=0; d<depth; ++d){
        this_d->SetToTensorLayer(this, d);
        T_d->SetToTensorLayer(T, d);
        this_d->Reverse(T_d);
    }
    DeleteOnlyShell(T_d);
    DeleteOnlyShell(this_d);
}

void tensor::SetDroppedColumnsToZero(activityData * activityColumns){
    if (!activityColumns->dropping)
        return;
    matrix* this_d = new matrix();
    for(int d=0; d<depth; ++d){
        this_d->SetToTensorLayer(this, d);
        this_d->SetDroppedElementsToZero(activityColumns);
    }
    DeleteOnlyShell(this_d);
}

void tensor::Rearrange(tensor * input){
    for(int d=0; d<input->depth; ++d)
        for(int r=0; r<input->rows; ++r)
            for(int c=0; c<input->cols; ++c)
                this->At(r, c, d) = input->At(d, r, c);
}


void tensor::BackwardRearrange(tensor * separatedInput){
    for(int d=0; d<depth; ++d)
        for(int r=0; r<rows; ++r)
            for(int c=0; c<cols; ++c)
                this->At(d, r, c) = separatedInput->At(r, c, d);
}
