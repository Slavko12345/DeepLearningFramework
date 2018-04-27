#ifndef __realNumber__
#define __realNumber__
#include "orderedData.h"
class vect;

struct realNumber: public orderedData{
    realNumber();
    realNumber(double val);
    int Dimensionality();
    double& At();
    void SetToVectElement(vect* V, int j);
};

#endif // __realNumber__
