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
    double *elem;
    int len;

    orderedData();
    orderedData(int len_);
    virtual ~orderedData();

    void AllocateMemory(int len_);

    void SetToZero();
    void SetToZeroStartingFrom(int startingIndex);
    void SetToValue(double val);
    void SetToRandomValues(double maxAbs);

    virtual void Print();

    void ReadFromFile(char filename[]);
    void ReadFromFile(ifstream &f);

    virtual void WriteToFile(char filename[]);
    void WriteToFile(ofstream &f);

    void Copy(orderedData* A);

    void CopyToSubpart(orderedData* A);

    void CopyMultiplied(double lambda, orderedData* A);

    void SetToLinearCombination(double a1, double a2, orderedData* A1, orderedData* A2);

    void Add(orderedData* addon);
    void Add(double lamb, orderedData* addon);
    void Add(double addon);

    void AddThisStartingFrom(int thisStartingIndex, orderedData* addon);

    void AddThisStartingFromOnlyActive(int thisStartingIndex, orderedData* addon, activityData* this_Activity);

    void AddAddonStartingFrom(int addonStartingIndex, orderedData* addon);

    void AddAddonStartingFromOnlyActive(int addonStartingIndex, orderedData* addon, activityData* this_Activity);

    double Sum();
    double SqNorm();

    void AddPointwiseFunction(orderedData* inp, mathFunc* func);
    void DroppedAddPointwiseFunction(orderedData* inp, mathFunc* func, activityData* inputActivity, activityData* outputActivity);

    void SetToReluFunction();
    void BackwardRelu(orderedData* input);
    void SetToMinReluFunction(orderedData* inp);
    void BranchMaxMin(orderedData* inp);

    void MaxMinBackward(orderedData* minOutputDelta, orderedData* output);

    void AddPointwiseFuncDerivMultiply(orderedData* inp, orderedData* funcArg, mathFunc* func);
    void DroppedAddPointwiseFuncDerivMultiply(orderedData* inp, orderedData* funcArg, mathFunc* func, activityData* inputActivity, activityData* outputActivity);

    void Multiply(double lamb);
    void PointwiseMultiply(orderedData* A);

    void MatrProd(matrix* M, vect* r);
    void TrMatrProd(matrix* M, vect* r);

    double Max();
    double Min();
    int ArgMax();
    double MaxAbs();

    void RmspropUpdate(orderedData* grad, orderedData* MS, double k1, double k2, double Step);
    void AdamUpdate(orderedData* grad, orderedData* Moment, orderedData* MS, double k1, double k2, double Step);

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

    void FindTrustRegionMinima(matrix* B, vect* r, double eps);
    void FindTrustRegionMinima(vect* eigenValues, matrix* eigenVectors, vect* rV, double trust_region_size);

    void CalculateMeanStdDev(double & mean, double & stDev);

    void NormalizeMeanStDev(orderedData* input, double & mean, double & stDev);

    void computeMedian(double * median, int * index);

    void computeMedianNonzero(double * median, int * index);

    void computeQuartiles(double * quartiles, int * index, int numQuartiles);

    void computeQuartilesNonzero(double * quartiles, int * index, int numQuartiles);

    void AverageWith(orderedData * input);

    void BackwardAverageWith(orderedData* inputDelta);
};

double InnerProduct(orderedData* inp1, orderedData* inp2);

double InnerProductSubMatrices(matrix* M1, matrix* M2, int border);
#endif // __orderedData__
