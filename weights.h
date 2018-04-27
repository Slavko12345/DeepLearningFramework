#ifndef __weights__
#define __weights__

class orderedData;
class matrix;

struct layerWeight{
    orderedData * dataWeight;
    orderedData * bias;

    void SetSymmetricConvolutionRelu(int startDepth, int numStairs, int numStairConvolutions, int symmetryLevel);

    void SetStairsFullConvolution(int startDepth, int numStairs, int numStairConvolutions, int symmetryLevel = 0, bool biasIncluded = 1);

    void SetStandard(int initialNumber, int nConvolutions);
    void SetStandardLinear(int startNumber, int nConvolutions);
    void SetStandardMultiple(int startDepth, int nConvolutions);
    void SetBottleneckStandardVert(int initialNumber, int nConvolutions);
    void SetBottleneckStandardHor(int nConvolutions);

    void SetBottleneckStandardReluVert(int initialNumber, int nConvolutions);
    void SetBottleneckStandardReluHor(int nConvolutions);

    void SetBottleneckStandardLimitedVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth);
    void SetBottleneckStandardLimitedHor(int nConvolutions);

    void SetBottleneckStandardRandomSymmetricVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth);
    void SetBottleneckStandardRandomSymmetricHor(int nConvolutions);

    void SetBottleneckStandardRandomSymmetricNoBiasVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth);
    void SetBottleneckStandardRandomSymmetricNoBiasHor(int nConvolutions);

    void SetBottleneckStandardRandomFullySymmetricVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth);
    void SetBottleneckStandardRandomFullySymmetricHor(int nConvolutions);

    void SetBottleneckStandardRandomLimitedVert(int startNumber, int nConvolutions, int limitDepth, int alwaysPresentDepth);
    void SetBottleneckStandardRandomLimitedHor(int nConvolutions);

    void SetStairsVert(int startDepth, int numStairs, int numStairConvolutions);
    void SetStairsHor(int numStairs, int numStairConvolutions);
    void SetStairsSymmHor(int numStairs, int numStairConvolutions, int symmetryLevel);

    void SetSequentialVert(int startDepth, int numStairs, int numStairConvolutions);
    void SetSequentialHor(int numStairs, int numStairConvolutions, int symmetryLevel);

    void SetFC(int fromSize, int toSize);
    void SetFCNoBias(int fromSize, int toSize);
};

class weights{
public:
    int Nweights;
    layerWeight * weightList;
    void SetModel();
    void SetToZero();
    void SetToRandomValues(double maxAbs);
    void Print();

    void Add(weights* addon);
    void Add(double lamb, weights* addon);

    double MaxAbs();
    void Multiply(double lamb);

    void RmspropUpdate(weights* grad, weights* MS, double k1, double k2, double Step);
    void AdamUpdate(weights* grad, weights* Moment, weights* MS, double k1, double k2, double Step);

    void Copy(weights* W);
    void CopyMultiplied(double lamb, weights* W);

    void WriteToFile(char fileName[]);
    void ReadFromFile(char filename[]);

    void SetToLinearCombination(double a1, double a2, weights* W1, weights* W2);

    int GetWeightLen();

    ~weights();
};

#endif // __weights__

//Wc=Wa+Wb*lambda
void AddWeights(weights* Wa, weights *Wb, double lambda, weights* Wc);

void FormOrthonormalBasis(weights* grad, weights* moment, weights* d1, weights* d2, double & g_g);

double InnerProduct(weights* w1, weights* w2);

void FormHessianInSubspace(weights* d1, weights* d2, weights* Hd1, weights* Hd2, matrix* B);