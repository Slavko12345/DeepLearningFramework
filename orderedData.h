#ifndef __orderedData__
#define __orderedData__
#include <vector>
#include <fstream>
using namespace std;

class mathFunc;
class activityData;
class matrix;
class vect;

struct orderedData{
    float *elem;
    int len;

    orderedData();
    orderedData(int len_);
    virtual ~orderedData();

    void AllocateMemory(int len_);

    void SetToZero();
    void SetToZeroStartingFrom(int startingIndex);
    void SetToValue(float val);
    void SetToRandomValues(float maxAbs);

    virtual void Print();

    void ReadFromFile(char filename[]);
    void ReadFromFile(ifstream &f);

    virtual void WriteToFile(char filename[]);
    void WriteToFile(ofstream &f);

    void Copy(orderedData* A);

    void CopyToSubpart(orderedData* A);

    void CopyMultiplied(float lambda, orderedData* A);

    void SetToLinearCombination(float a1, float a2, orderedData* A1, orderedData* A2);

    void Add(orderedData* addon);
    void Add(float lamb, orderedData* addon);
    void Add(float addon);

    void AddDistinct(orderedData* addon);
    void AddDistinct(float lamb, orderedData* addon);
    void AddDistinct_1024(float lamb, orderedData* addon);
    void AddDistinct_256(float lamb, orderedData* addon);
    void AddDistinct_64(float lamb, orderedData* addon);


    void AddThisStartingFrom(int thisStartingIndex, orderedData* addon);

    void AddThisStartingFromOnlyActive(int thisStartingIndex, orderedData* addon, activityData* this_Activity);

    void AddAddonStartingFrom(int addonStartingIndex, orderedData* addon);

    void AddAddonStartingFromOnlyActive(int addonStartingIndex, orderedData* addon, activityData* this_Activity);

    float Sum();
    float SqNorm();
    float Mean();

    void AddPointwiseFunction(orderedData* inp, mathFunc* func);
    void DroppedAddPointwiseFunction(orderedData* inp, mathFunc* func, activityData* inputActivity, activityData* outputActivity);

    void SetToReluFunction();
    void BackwardRelu(orderedData* input);
    void SetToMinReluFunction(orderedData* inp);
    void BranchMaxMin(orderedData* inp);

    void MaxMinBackward(orderedData* minOutputDelta, orderedData* output);

    void AddPointwiseFuncDerivMultiply(orderedData* inp, orderedData* funcArg, mathFunc* func);
    void DroppedAddPointwiseFuncDerivMultiply(orderedData* inp, orderedData* funcArg, mathFunc* func, activityData* inputActivity, activityData* outputActivity);

    void Multiply(float lamb);
    void PointwiseMultiply(orderedData* A);

    void MatrProd(matrix* M, vect* r);
    void TrMatrProd(matrix* M, vect* r);

    float Max();
    float Min();
    int ArgMax();
    float MaxAbs();

    void RmspropUpdate(orderedData* grad, orderedData* MS, float k1, float k2, float Step);
    void AdamUpdate(orderedData* grad, orderedData* Moment, orderedData* MS, float k1, float k2, float Step);

    void SetDroppedElementsToZero(activityData* mask);

    void SetDroppedElementsToZero(activityData* mask, int maxLen);

    void SetDroppedElementsToZero(activityData* mask, int startingInd, int maxLen);

    void ListNonzeroElements(vector<int> & index);
    void ListNonzeroElements(vector<int> & index, orderedData* compressed);

    void ListNonzeroActiveElements(vector<int> & index, activityData* activity);
    void ListNonzeroActiveElements(vector<int> & index, activityData* activity, orderedData* compressed);

    void ListActiveElements(vector<int> & index, activityData* activity);
    void ListActiveElements(vector<int> & index, activityData* activity, orderedData* compressed);

    void ListAll(vector<int> & index);
    void ListAll(vector<int> & index, orderedData* compressed);

    void AddMatrVectProductBias(matrix* kernel, orderedData* input, vect* bias, vector<int> &indexInput, vector<int> &indexOutput);
    void AddMatrVectProductBias(matrix* kernel, orderedData* input, vect* bias, vector<int> &indexInput);
    void AddMatrVectProductBias(matrix* kernel, orderedData* input, vect* bias);
    void AddMatrVectProduct(matrix* kernel, orderedData* input);

    void AddMatrCompressedVectProductBias(matrix* kernel, orderedData* compressedInput, vect* bias, vector<int> &indexInput);

    void BackwardFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad, vect* biasGrad,
                                  vector<int> &indexInput, vector<int> &indexOutput);

    void BackwardFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad, vect* biasGrad,
                                  vector<int> &indexInput);

    void BackwardFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad, vect* biasGrad);

    void BackwardFullyConnectedNoBias(matrix* kernel, orderedData* outputDelta, orderedData* input, matrix* kernelGrad);

    void BackwardCompressedInputFullyConnected(matrix* kernel, orderedData* outputDelta, orderedData* compressedInput, matrix* kernelGrad, vect* biasGrad,
                                  vector<int> &indexInput);

    virtual int Dimensionality() = 0;

    void FindTrustRegionMinima(matrix* B, vect* r, float eps);
    void FindTrustRegionMinima(vect* eigenValues, matrix* eigenVectors, vect* rV, float trust_region_size);

    void CalculateMeanStdDev(float & mean, float & stDev);

    void NormalizeMeanStDev(orderedData* input, float & mean, float & stDev);

    void computeMedian(float * median, int * index);

    void computeMedianNonzero(float * median, int * index);

    void computeQuartiles(float * quartiles, int * index, int numQuartiles);

    void computeQuartilesNonzero(float * quartiles, int * index, int numQuartiles);

    void AverageWith(orderedData * input);

    void BackwardAverageWith(orderedData* inputDelta);

    void SetToBalancedMultipliers(activityData* balancedActiveUnits, activityData* balacedUpDown, float alpha);
};

float InnerProduct(orderedData* inp1, orderedData* inp2);

float InnerDistinctProduct(orderedData* inp1, orderedData* inp2);

float InnerDistinctProduct_1024(orderedData* inp1, orderedData* inp2);
float InnerDistinctProduct_256(orderedData* inp1, orderedData* inp2);
float InnerDistinctProduct_64(orderedData* inp1, orderedData* inp2);


float InnerProductSubMatrices(matrix* M1, matrix* M2, int border);
#endif // __orderedData__
