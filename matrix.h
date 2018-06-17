#ifndef __matrix__
#define __matrix__

#include "orderedData.h"
#include <vector>
using namespace std;

class mathFunc;
class vect;
class activityMatrix;
class tensor;

struct matrix: public orderedData
{
    int rows;
    int cols;
    matrix();
    matrix(int rows_, int cols_);
    void SetSize(int rows_, int cols_);
    void PointToMatrix(int rows_, int cols_, float* elem_);
    void PointToMatrix(float *elem_);

    void Print();
    void WriteToFile(char filename[]);

    float& At(int r, int c);
    int Ind(int r, int c);
    int IndRow(int r);
    float *Row(int r);


    void SaveAsImage(char filename[]);
    void SaveNormalizedAsImage(char filename[]);

    void BackwardFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* input, vect* biasGrad, vector<int>& indexOutput, vector<int>& indexInput);
    void BackwardFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* input, vect* biasGrad, vector<int>& indexInput);
    void BackwardCompressedInputFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* compressedInput, vect* biasGrad, vector<int>& indexInput);
    void BackwardFullyConnectedOnlyGrad(orderedData* outputDelta, orderedData* input, vect* biasGrad);
    void BackwardFullyConnectedNoBiasOnlyGrad(orderedData* outputDelta, orderedData* input);
    //void AddMatrMatrProduct(matrix* A, matrix* B);
    int Dimensionality();

    void SetToTensorLayer(tensor* T, int d);

    void Reverse(matrix* M);

    void SubMatrix(matrix* M, int rows_);

    void SubMatrix(matrix* M, int startRow, int rows_);

    void EigenDecompose(vect* eigenValues, matrix* eigenVectors);

    float Interpolate(float x, float y);

    void CopySubMatrixMultiplied(float lamb, matrix* M, int border);
    void AddSubMatrix(float lamb, matrix* M, int border);
};

#endif
