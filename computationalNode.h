#ifndef __computationalNode__
#define __computationalNode__
#include <vector>
using namespace std;
class layers;
class weights;
class matrix;
class realNumber;
class vect;
class tensor;
class tensor4D;
class orderedData;
class mathFunc;
class activityLayers;
class activityData;


struct computationalNode{
    activityData* inputActivity;
    activityData* outputActivity;
    bool testMode;
    bool primalWeight;

    computationalNode();
    virtual void ForwardPass()=0;
    virtual void BackwardPass(bool computeDelta, int trueClass)=0;
    virtual bool HasWeightsDependency()=0;
    virtual void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    virtual void SetToTrainingMode();
    virtual void SetToTestMode();
    virtual bool NeedsUnification();
    virtual void Unify(computationalNode * primalCN);
    virtual void WriteStructuredWeightsToFile();
    virtual ~computationalNode();
};


struct FullyConnected: public computationalNode{
    int weightsNum;
    matrix* kernel, *kernelGrad;
    vect* bias, *biasGrad;
    orderedData* input, * output;
    orderedData* inputDelta, *outputDelta;
    vector<int> indexInput;
    vector<int> indexOutput;

    FullyConnected(int weightsNum_);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    ~FullyConnected();
};


struct Convolution2D2D: public computationalNode{
    int weightsNum;
    matrix* input, *inputDelta;
    matrix* output, *outputDelta;
    matrix* kernel, *kernelGrad, *reversedKernel;
    realNumber*   bias, *biasGrad;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    Convolution2D2D(int weightsNum_, int paddingR_, int paddingC_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~Convolution2D2D();
};

struct Convolution2D3D: public computationalNode{
    int weightsNum;
    matrix* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* kernel, *kernelGrad, *reversedKernel;
    vect*   bias, *biasGrad;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    Convolution2D3D(int weightsNum_, int paddingR_, int paddingC_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~Convolution2D3D();
};

struct Convolution3D2D: public computationalNode{
    int weightsNum;
    tensor* input, *inputDelta;
    matrix* output, *outputDelta;
    tensor* kernel, *kernelGrad, *reversedKernel;
    realNumber*   bias, *biasGrad;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    Convolution3D2D(int weightsNum_, int paddingR_, int paddingC_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~Convolution3D2D();
};

struct Convolution3D3D: public computationalNode{
    int weightsNum;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor4D* kernel, *kernelGrad, *reversedKernel;
    vect*   bias, *biasGrad;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    Convolution3D3D(int weightsNum_, int paddingR_, int paddingC_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~Convolution3D3D();
};

struct PointwiseFunctionLayer: public computationalNode{
    mathFunc* func;
    orderedData* input, *inputDelta;
    orderedData* output, *outputDelta;

    PointwiseFunctionLayer(mathFunc* func_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
};

struct SoftMaxLayer: public computationalNode{
    vect* input;
    vect* output;

    SoftMaxLayer();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
};


struct MaxPooling2D: public computationalNode{
    int *rowInd, *colInd;
    matrix* input, *inputDelta;
    matrix* output, *outputDelta;
    int kernelRsize, kernelCsize;

    MaxPooling2D(int kernelRsize_, int kernelCsize_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxPooling2D();
};


struct MaxPooling3D: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    int kernelRsize, kernelCsize;

    MaxPooling3D(int kernelRsize_, int kernelCsize_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxPooling3D();
};


struct MaxAbsPooling3D: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* maxAbs;
    int kernelRsize, kernelCsize;

    MaxAbsPooling3D(int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxAbsPooling3D();
};


struct PartialMaxAbsPooling3D: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* maxAbs;
    tensor* partialInput, *partialOutput;
    tensor* partialInputDelta, *partialOutputDelta;
    int lastLayers;
    int startingDepth;
    int kernelRsize, kernelCsize;

    PartialMaxAbsPooling3D(int lastLayers_, int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~PartialMaxAbsPooling3D();
};

struct FullMaxAbsPooling: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* maxAbs;
    tensor* partialOutput;
    tensor *partialOutputDelta;
    int numLayers;
    int kernelRsize, kernelCsize;

    FullMaxAbsPooling();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~FullMaxAbsPooling();
};

struct LastAveragePooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput, *partialOutput;
    tensor* partialInputDelta, *partialOutputDelta;
    int lastLayers;
    int startingDepth;
    int kernelRsize, kernelCsize;
    LastAveragePooling(int lastLayers_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~LastAveragePooling();
};


struct StructuredDropAveragePooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput;
    tensor* partialInputDelta;
    activityData * activityColumns;

    int lastLayers;
    int startingDepth;
    double dropRate;
    int activityLen;

    StructuredDropAveragePooling(double dropRate_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass(bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~StructuredDropAveragePooling();
};



struct ColumnDrop: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput;
    tensor* partialInputDelta;
    activityData * activityColumns;

    int lastLayers;
    int remainNum;

    ColumnDrop(int remainNum_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass(bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~ColumnDrop();
};



struct StructuredDropAverageSubPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput;
    tensor* partialInputDelta;
    activityData * activityColumns;

    int lastLayers;
    int startingDepth;
    int border;
    double dropRate;


    StructuredDropAverageSubPooling(int border_, double dropRate_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~StructuredDropAverageSubPooling();
};



struct PartialFirstAveragePooling3D: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput, *partialOutput;
    tensor* partialInputDelta, *partialOutputDelta;

    int startingOutputDepth;
    int firstLayers;

    int kernelRsize, kernelCsize;
    PartialFirstAveragePooling3D(int startingOutputDepth_, int firstLayers_, int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialFirstAveragePooling3D();
};



struct FullAveragePooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialOutput, *partialOutputDelta;

    int kernelRsize, kernelCsize;

    FullAveragePooling();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~FullAveragePooling();
};


struct PartialSubPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput, *partialInputDelta;

    int border;
    int lastLayers;

    PartialSubPooling(int border_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialSubPooling();
};




struct PartialMeanVarPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor *partialOutput;
    tensor *partialOutputDelta;

    int startingOutputDepth;

    PartialMeanVarPooling(int startingOutputDepth_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialMeanVarPooling();
};


struct CenterPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    vect* sum;

    CenterPooling();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~CenterPooling();
};


struct MedianPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    int * index;

    MedianPooling();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~MedianPooling();
};


struct QuartilesPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    int * index;
    int numQuartiles;

    QuartilesPooling(int numQuartiles_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~QuartilesPooling();
};




struct AverageMaxAbsPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor *partialOutput;
    tensor *partialOutputDelta;

    int onlyAveragePoolingDepth;
    int startingOutputDepth;
    int kernelRsize;
    int kernelCsize;

    int * rowInd;
    int * colInd;

    AverageMaxAbsPooling(int onlyAveragePoolingDepth_ = 3, int startingOutputDepth_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~AverageMaxAbsPooling();
};


struct PartialMeanQuadStatsPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor *partialOutput;
    tensor *partialOutputDelta;

    int startingOutputDepth;

    PartialMeanQuadStatsPooling(int startingOutputDepth_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialMeanQuadStatsPooling();
};


struct PartialMeanStDevPooling: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor *partialOutput;
    tensor *partialOutputDelta;

    int startingOutputDepth;

    PartialMeanStDevPooling(int startingOutputDepth_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialMeanStDevPooling();
};


struct PartialFirstLastAveragePooling3D: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialFirstInput, *partialFirstOutput;
    tensor* partialFirstInputDelta, *partialFirstOutputDelta;
    tensor* partialLastInput, *partialLastOutput;
    tensor* partialLastInputDelta, *partialLastOutputDelta;

    int startingOutputDepth;
    int firstLayers;
    int lastLayers;

    int kernelRsize, kernelCsize;
    PartialFirstLastAveragePooling3D(int startingOutputDepth_, int firstLayers_, int lastLayers_, int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialFirstLastAveragePooling3D();
};


struct PartialMiddleAveragePooling3D: public computationalNode{
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* partialInput, *partialOutput;
    tensor* partialInputDelta, *partialOutputDelta;

    int startingInputDepth;
    int numLayers;
    int startingOutputDepth;

    int kernelRsize, kernelCsize;
    PartialMiddleAveragePooling3D(int startingInputDepth_, int numLayers_, int startingOutputDepth_, int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~PartialMiddleAveragePooling3D();
};



struct MaxMinPooling3D: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* maxOut, *minOut, *maxOutDelta, *minOutDelta;
    int kernelRsize, kernelCsize;

    MaxMinPooling3D(int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxMinPooling3D();
};



struct MaxMinPoolingIndex3D: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* maxOut, *minOut, *maxOutDelta, *minOutDelta;
    int kernelRsize, kernelCsize;

    MaxMinPoolingIndex3D(int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxMinPoolingIndex3D();
};


struct MaxAbsPoolingIndex3D: public computationalNode{
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor* maxAbs;
    int kernelRsize, kernelCsize;

    MaxAbsPoolingIndex3D(int kernelRsize_ = 2, int kernelCsize_ = 2);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxAbsPoolingIndex3D();
};


struct MaxAbsPoolingSoftIndex3D: public computationalNode{
    double logFactor;
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    vect* output, *outputDelta;
    vect* maxAbs;
    tensor* softMaxInput;

    MaxAbsPoolingSoftIndex3D(double logFactor_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxAbsPoolingSoftIndex3D();
};



struct MaxAbsPoolingSoftDiffIndex3D: public computationalNode{
    double logFactor;
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    vect* output, *outputDelta;
    vect* maxAbs;
    tensor* softMaxInput;
    vect* tempOutput;
    vect* tempOutputDelta;

    MaxAbsPoolingSoftDiffIndex3D(double logFactor_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~MaxAbsPoolingSoftDiffIndex3D();
};



struct PartialMaxAbsPoolingSoftDiffIndex3D: public computationalNode{
    double logFactor;
    int lastLayers;
    int startingDepth;
    int *rowInd, *colInd;
    tensor* input, *inputDelta;
    vect* output, *outputDelta;

    tensor* partialInput, *partialInputDelta;
    vect* partialOutput, *partialOutputDelta;

    vect* maxAbs;
    tensor* softMaxInput;
    vect* tempOutput;
    vect* tempOutputDelta;

    PartialMaxAbsPoolingSoftDiffIndex3D(int lastLayers_, double logFactor_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~PartialMaxAbsPoolingSoftDiffIndex3D();
};




//struct MaxMinPoolingSoftMaxIndex3D: public computationalNode{
//    int *rowInd, *colInd;
//    tensor* input, *inputDelta;
//    vect* output, *outputDelta;
//    vect* maxOut, *minOut, *maxOutDelta, *minOutDelta;
//    int kernelRsize, kernelCsize;
//
//    MaxMinPoolingSoftMaxIndex3D();
//    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to);
//    void ForwardPass();
//    void BackwardPass( bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//    void BackpropagateDroppedUnits();
//    ~MaxMinPoolingSoftMaxIndex3D();
//};



struct AveragePooling2D: public computationalNode{
    int kernelRsize, kernelCsize;
    matrix* input, *inputDelta;
    matrix* output, *outputDelta;

    AveragePooling2D(int kernelRsize_, int kernelCsize_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
};


struct AveragePooling3D: public computationalNode{
    int kernelRsize, kernelCsize;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;

    AveragePooling3D(int kernelRsize_, int kernelCsize_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
};


struct Merge: public computationalNode{
    int startingIndex;
    orderedData* input, *inputDelta;
    orderedData* output, *outputDelta;

    Merge(int startingIndex_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
};

struct ConvoluteReluMerge: public computationalNode{
    int weightsNum;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor4D* kernel, *kernelGrad, *reversedKernel;
    tensor* convolutionOutput, *convolutionOutputDelta;
    vect*   bias, *biasGrad;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    ConvoluteReluMerge(int weightsNum_, int paddingR_, int paddingC_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~ConvoluteReluMerge();
};


struct ConvoluteMaxMinMerge: public computationalNode{
    int weightsNum;
    int startingIndexMerge;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;
    tensor4D* kernel, *kernelGrad, *reversedKernel;
    vect*   bias, *biasGrad;
    tensor* convolutionOutput, *minOutput;
    tensor* convolutionOutputDelta, * minOutputDelta;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    ConvoluteMaxMinMerge(int weightsNum_, int paddingR_ = 1, int paddingC_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~ConvoluteMaxMinMerge();
};


struct SequentiallyConvoluteMaxMin: public computationalNode{
    int weightsNum;
    int startDepth;
    int paddingR, paddingC;
    tensor* input, *inputDelta;
    tensor* kernel, *kernelGrad, *reversedKernel;
    vect* bias, *biasGrad;
    vector<int> indexInputRow, indexOutputRow;

    SequentiallyConvoluteMaxMin(int weightsNum_, int startDepth_, int paddingR_ = 1, int paddingC_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~SequentiallyConvoluteMaxMin();
};

struct SequentiallyConvoluteMaxMinStandard: public computationalNode{
    int weightsNum;
    int startDepth;
    tensor* input, *inputDelta;
    tensor* kernel, *kernelGrad;
    vect* bias, *biasGrad;

    SequentiallyConvoluteMaxMinStandard(int weightsNum_, int startDepth_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~SequentiallyConvoluteMaxMinStandard();
};

struct SequentiallyConvoluteBottleneckReluStandard: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    SequentiallyConvoluteBottleneckReluStandard(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~SequentiallyConvoluteBottleneckReluStandard();
};


struct SequentiallyConvoluteBottleneckMaxMinStandard: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    bool firstLayer;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    SequentiallyConvoluteBottleneckMaxMinStandard(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~SequentiallyConvoluteBottleneckMaxMinStandard();
};



struct SequentiallyMaxMinStandard: public computationalNode{
    int weightsNum;
    int startDepth;
    int nConvolutions;

    tensor* input, *inputDelta;
    vect* kernel, *kernelGrad;
    vect* bias, *biasGrad;

    SequentiallyMaxMinStandard(int weightsNum_, int startDepth_, int nConvolutions_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~SequentiallyMaxMinStandard();
};





struct SequentiallyConvoluteBottleneckMaxMinStandardLimited: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    int limitDepth;
    int alwaysPresentDepth;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    SequentiallyConvoluteBottleneckMaxMinStandardLimited(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    ~SequentiallyConvoluteBottleneckMaxMinStandardLimited();
};



struct SequentiallyConvoluteBottleneckMaxMinStandardRandom: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    int limitDepth;
    int alwaysPresentDepth;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    SequentiallyConvoluteBottleneckMaxMinStandardRandom(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    ~SequentiallyConvoluteBottleneckMaxMinStandardRandom();
};



struct SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    int limitDepth;
    int alwaysPresentDepth;

    bool innerDropping;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;
    activityData * verticalConvActivity;

    SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric(int weightsNum_vertical_, int weightsNum_horizontal_,
                    int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_, bool innerDropping_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    ~SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetric();
};



struct SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    int limitDepth;
    int alwaysPresentDepth;

    bool innerDropping;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;

    tensor * verticalConv, * verticalConvDelta;
    activityData * verticalConvActivity;

    SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias(int weightsNum_vertical_, int weightsNum_horizontal_,
                    int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_, bool innerDropping_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~SequentiallyConvoluteBottleneckMaxMinStandardRandomSymmetricNoBias();
};




//symmetric kernel
//no bias
//bottleneck layers
//max-min nonlinearity
struct StairsConvolution: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    bool innerDropping;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;

    tensor * verticalConv, * verticalConvDelta;
    activityData * verticalConvActivity;

    StairsConvolution(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_, bool innerDropping_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsConvolution();
};


struct StairsSymmetricConvolution: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    int symmetryLevel;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    StairsSymmetricConvolution(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsSymmetricConvolution();
};


struct StairsSymmetricConvolutionRelu: public computationalNode{
    //RELU units
    //no bottleneck - full layers
    int weightsNum;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    int symmetryLevel;

    tensor* input, *inputDelta;
    tensor* kernel, *kernelGrad;

    StairsSymmetricConvolutionRelu(int weightsNum_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsSymmetricConvolutionRelu();
};


struct StairsFullConvolution: public computationalNode{
    //Max-Min units
    //no bottleneck - full layers
    int weightsNum;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    int symmetryLevel;
    bool biasIncluded;

    tensor* input, *inputDelta;
    tensor* kernel, *kernelGrad;
    vect* bias, *biasGrad;

    StairsFullConvolution(int weightsNum_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_ = 0, bool biasIncluded_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsFullConvolution();
};



struct StairsFullConvolutionRelu: public computationalNode{
    //RELU units
    //no bottleneck - full layers
    int weightsNum;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    int symmetryLevel;
    bool biasIncluded;

    tensor* input, *inputDelta;
    tensor* kernel, *kernelGrad;
    vect* bias, *biasGrad;

    StairsFullConvolutionRelu(int weightsNum_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_ = 0, bool biasIncluded_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsFullConvolutionRelu();
};




struct StairsSequentialConvolution: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    int symmetryLevel;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    StairsSequentialConvolution(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_, int symmetryLevel_ = 0);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsSequentialConvolution();
};





struct StairsPyramidalConvolution: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int numStairs;
    int numStairConvolutions;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    StairsPyramidalConvolution(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int numStairs_, int numStairConvolutions_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    void WriteStructuredWeightsToFile();
    ~StairsPyramidalConvolution();
};





struct SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    int limitDepth;
    int alwaysPresentDepth;

    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    ~SequentiallyConvoluteBottleneckMaxMinStandardRandomFullySymmetric();
};






struct SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited: public computationalNode{
    int weightsNum_vertical;
    int weightsNum_horizontal;

    int startDepth;
    int nConvolutions;
    int limitDepth;
    int alwaysPresentDepth;


    int * indices;

    tensor* input, *inputDelta;
    tensor* kernel_vert, *kernelGrad_vert;
    tensor* kernel_hor, *kernelGrad_hor;
    vect* bias_hor, *biasGrad_hor;

    tensor * verticalConv, * verticalConvDelta;

    SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited(int weightsNum_vertical_, int weightsNum_horizontal_, int startDepth_, int nConvolutions_, int limitDepth_, int alwaysPresentDepth_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    bool NeedsUnification();
    void Unify(computationalNode * primalCN);
    ~SequentiallyConvoluteBottleneckMaxMinStandardRandomLimited();
};





struct SequentiallyConvoluteMultipleMaxMinStandard: public computationalNode{
    int weightsNum;
    int startDepth;
    int nConvolutions;
    tensor* input, *inputDelta;
    tensor* kernel, *kernelGrad;
    vect* bias, *biasGrad;

    SequentiallyConvoluteMultipleMaxMinStandard(int weightsNum_, int startDepth_, int nConvolutions_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~SequentiallyConvoluteMultipleMaxMinStandard();
};



struct ConvoluteDependentMaxMinMerge: public computationalNode{
    int weightsNum;
    int paddingR, paddingC;
    int poolingKernelR, poolingKernelC;

    tensor* input, *inputDelta;
    tensor* output, *outputDelta;

    tensor* pooledInput, *pooledInputDelta;

    tensor4D* convolutionKernel, *convolutionKernelGrad, *convolutionReversedKernel;
    vect*   convolutionBias, *convolutionBiasGrad;

    matrix* FCkernel, *FCkernelGrad;
    vect* FCbias, *FCbiasGrad;

    matrix* FCkernelCkernel, *FCkernelCkernelGrad;
    matrix* FCkernelCbias, *FCkernelCbiasGrad;

    vect* FCbiasCkernel, *FCbiasCkernelGrad;
    vect* FCbiasCbias, *FCbiasCbiasGrad;

    tensor* inputAddonDelta;


    tensor* convolutionOutput, *minOutput;
    tensor* convolutionOutputDelta, * minOutputDelta;

    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    vector<int> indexPooledInput;
    vector<int> indexConvolutionKernel;
    vector<int> indexConvolutionBias;

    ConvoluteDependentMaxMinMerge(int weightsNum_, int paddingR_ = 1, int paddingC_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~ConvoluteDependentMaxMinMerge();
};


struct ConvoluteQuadraticMaxMinMerge: public computationalNode{
    int weightsNum;
    int startingIndexMerge;
    tensor* input, *inputDelta;
    tensor* output, *outputDelta;

    tensor4D* kernel, *kernelGrad, *reversedKernel;
    tensor4D* linearKernel, *linearKernelGrad, *linearReversedKernel;
    tensor4D* quadraticKernel, *quadraticKernelGrad, *quadraticReversedKernel;

    vect*   bias, *biasGrad;
    tensor* convolutionOutput, *minOutput;
    tensor* convolutionOutputDelta, * minOutputDelta;
    int paddingR, paddingC;
    vector<int> indexInputRow;
    vector<int> indexOutputRow;

    ConvoluteQuadraticMaxMinMerge(int weightsNum_, int paddingR_ = 1, int paddingC_ = 1);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~ConvoluteQuadraticMaxMinMerge();
};




struct FCMaxMinMerge: public computationalNode{
    int weightsNum;
    int startingIndexMerge;
    matrix* kernel, *kernelGrad;
    vect* bias, *biasGrad;
    vect* input, * output;
    vect* inputDelta, *outputDelta;
    vector<int> indexInput;
    vector<int> indexOutput;
    vect* FCOutput, *minOutput;
    vect* FCOutputDelta, *minOutputDelta;

    FCMaxMinMerge(int weightsNum_);
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    bool HasWeightsDependency();
    ~FCMaxMinMerge();
};


struct FullyConnectedSoftMax: public computationalNode{
    int weightsNum;
    matrix* kernel, *kernelGrad;
    vect* bias, *biasGrad;
    orderedData* input, * output;
    orderedData* inputDelta, *outputDelta;
    //orderedData* compressedInput;
    //vector<int> indexInput;

    FullyConnectedSoftMax(int weightsNum_);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void WriteStructuredWeightsToFile();
    ~FullyConnectedSoftMax();
};


struct FullyConnectedNoBiasSoftMax: public computationalNode{
    int weightsNum;
    matrix* kernel, *kernelGrad;
    orderedData* input, * output;
    orderedData* inputDelta, *outputDelta;
    //orderedData* compressedInput;
    //vector<int> indexInput;

    FullyConnectedNoBiasSoftMax(int weightsNum_);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void WriteStructuredWeightsToFile();
    ~FullyConnectedNoBiasSoftMax();
};



struct Ensemble: public computationalNode{
    int weightsNum;
    int lastLayers;
    matrix* kernel, *kernelGrad;
    tensor* input, * inputDelta;
    orderedData* output;
    tensor* partialInput, *partialInputDelta;

    tensor* separateInput, *separateInputDelta;
    tensor* separateOutput, *separateOutputDelta;
    tensor* pooledInput;

    matrix * separateInput_r;
    matrix* separateInputDelta_r;
    matrix * separateOutput_r;
    matrix * separateOutputDelta_r;
    vect* separateInput_rc;
    vect* separateInputDelta_rc;
    vect* separateOutput_rc;
    vect* separateOutputDelta_rc;

    Ensemble(int weightsNum_, int lastLayers_);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void WriteStructuredWeightsToFile();
    ~Ensemble();
};



struct SymmetricEnsemble: public computationalNode{
    int weightsNum;
    int lastLayers;
    matrix* kernel, *kernelGrad;
    tensor* input, * inputDelta;
    orderedData* output;
    tensor* partialInput, *partialInputDelta;

    tensor* separateInput, *separateInputDelta;
    tensor* separateOutput, *separateOutputDelta;
    tensor* pooledInput;

    matrix * separateInput_r;
    matrix* separateInputDelta_r;

    matrix * separateOutput_r;
    matrix * separateOutputDelta_r;

    vect* separateInput_rc;
    vect* separateInputDelta_rc;
    vect* separateInput_r_1_c;
    vect* separateInputDelta_r_1_c;

    vect* separateOutput_rc;
    vect* separateOutputDelta_rc;

    SymmetricEnsemble(int weightsNum_, int lastLayers_);
    void ForwardPass();
    void BackwardPass( bool computeDelta, int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    bool HasWeightsDependency();
    void Initiate(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, int from, int to, bool primalWeightOwner);
    void WriteStructuredWeightsToFile();
    ~SymmetricEnsemble();
};




//
//struct Merge2D: public computationalNode{
//    int startingIndex;
//    Merge2D(int startingIndex_);
//    void ForwardPass(int from, int to);
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//    void BackpropagateDroppedUnits(int from, int to);
//};
//
//
//struct Merge3D: public computationalNode{
//    int startingIndex;
//    Merge3D(int startingIndex_);
//    void ForwardPass(int from, int to);
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//    void BackpropagateDroppedUnits(int from, int to);
//};
//

//struct TranslationInvariant: public computationalNode{
//    int maxDist;
//    matrix* res1;
//    matrix* res2;
//    matrix* res3;
//    TranslationInvariant(int maxDist_);
//    void ForwardPass(int from, int to);
//    //backward pass not implemented because of inefficiency
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//};
//
//struct CenterAtTheMean2D: public computationalNode{
//    int rowShift;
//    int colShift;
//    CenterAtTheMean2D();
//    void ForwardPass(int from, int to);
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//};
//
//struct CenterAtTheMean3D: public computationalNode{
//    int rowShift;
//    int colShift;
//    CenterAtTheMean3D();
//    void ForwardPass(int from, int to);
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//};
//

//
//struct Flatten2D: public computationalNode{
//    Flatten2D();
//    void ForwardPass(int from, int to);
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//    void BackpropagateDroppedUnits(int from, int to);
//};
//
//struct Flatten3D: public computationalNode{
//    Flatten3D();
//    void ForwardPass(int from, int to);
//    void BackwardPass(int from, int to, bool computeDelta, int trueClass);
//    bool HasWeightsDependency();
//    void BackpropagateDroppedUnits(int from, int to);
//};


#endif // __computationalNode__
