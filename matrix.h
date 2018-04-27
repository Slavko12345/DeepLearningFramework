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
    void PointToMatrix(int rows_, int cols_, double* elem_);
    void PointToMatrix(double *elem_);

    void Print();
    void WriteToFile(char filename[]);

    double& At(int r, int c);
    int Ind(int r, int c);
    int IndRow(int r);
    double *Row(int r);


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

    double Interpolate(double x, double y);

    void CopySubMatrixMultiplied(double lamb, matrix* M, int border);
    void AddSubMatrix(double lamb, matrix* M, int border);
};

#endif
