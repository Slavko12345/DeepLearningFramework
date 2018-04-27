#ifndef __tensor__
#define __tensor__

#include "orderedData.h"
#include <vector>
using namespace std;
class matrix;
class activityTensor;
class tensor4D;

class tensor: public orderedData
{
public:
    int depth;
    int rows;
    int cols;

    tensor();
    tensor(int depth_, int rows_, int cols_);
    void SetSize(int depth_, int rows_, int cols_);
    void PointToTensor(double * elem_);

    void Print();
    double &At(int d, int r, int c);
    int   Ind(int d, int r, int c);
    int   Ind(int d);
    double *Layer(int d);

    void SaveAsImage(char filename[]);

    void TransformByPoints(tensor* inputImage, matrix* coordsPoints);

    void RandomlyTranform(tensor* inputImage);

    int Dimensionality();

    void SetToTLayer(tensor4D* T4D, int n);

    void SubTensor(tensor* T, int depth_);

    void SubTensor(tensor* T, int startDepth, int depth_);

    void SubLastTensor(tensor* T, int lastLayers);

    void Reverse(tensor* T);

    void SetDroppedColumnsToZero(activityData * activityColumns);

    void Rearrange(tensor * input);

    void BackwardRearrange(tensor * separatedInput);
};
#endif // __tensor__
