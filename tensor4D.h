#ifndef __tensor4D__
#define __tensor4D__
#include "orderedData.h"

class tensor;
class tensor4D: public orderedData
{
public:
    int number;
    int depth;
    int rows;
    int cols;

    tensor4D();
    tensor4D(int number_, int depth_, int rows_, int cols_);

    void Print();

    float * TLayer(int number_);
    float& At(int n, int d, int r, int c);
    int   Ind(int n, int d, int r, int c);

    int Dimensionality();

    void Reverse(tensor4D* T4D);
    void Sub4DTensor(tensor4D* T4D, int number_);
    void Sub4DTensor(tensor4D* T4D, int startingNumber, int number_);

};

#endif // __tensor4D__
