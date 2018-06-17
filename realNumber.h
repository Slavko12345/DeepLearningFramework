#ifndef __realNumber__
#define __realNumber__
#include "orderedData.h"
class vect;

struct realNumber: public orderedData{
    realNumber();
    realNumber(float val);
    int Dimensionality();
    float& At();
    void SetToVectElement(vect* V, int j);
};

#endif // __realNumber__
