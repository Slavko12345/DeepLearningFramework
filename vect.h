#ifndef __vect__
#define __vect__

#include "orderedData.h"
#include <vector>
using namespace std;

class matrix;
class activityVect;
class realNumber;


struct vect: public orderedData
{
    vect();
    vect(int len_);

    float & operator[](int index);
    const float &operator[](int index) const;

    void StaticCastToVect(orderedData* arg);

    void Add(vect* addon, vector<int> &indexOutput);

    //void AddTrMatrVectProduct(matrix* A, vect* B, vector<int> &toBeComputed, vector<int> &bIndices);

    int Dimensionality();

    void SetToMatrixRow(matrix* M, int r);

    float * elemLink(int j);

    void SubVect(vect* V, int len_);
    void SubVect(vect* V, int startInd, int len_);


};

#endif // __vect__
