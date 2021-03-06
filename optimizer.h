#ifndef __optimiser__
#define __optimiser__

class NeuralNet;
class Data;

class Optimizer{
public:
    int maxEpochs;
    //int maxIterations;
    int mbSize;
    Optimizer(int maxEpochs_, int mbSize_);
    virtual void Optimize(NeuralNet* NN, Data* trainingData)=0;
    virtual void OptimizeInParallel(NeuralNet *NN, Data* trainingData);
    virtual void OptimizeInParallel(NeuralNet *NN, Data* trainingData, Data* testData);
    virtual ~Optimizer();
};

class SGD: public Optimizer{
public:
    float learningRate;
    SGD(float learningRate_, int mbSize_, int maxEpochs_);
    void Optimize(NeuralNet* NN, Data* trainingData);
};

class RMSPROP: public Optimizer{
public:
    float learningRate;
    RMSPROP(float learningRate_, int mbSize_, int maxEpochs_);
    void Optimize(NeuralNet* NN, Data* trainingData);
};

class ADAM: public Optimizer{
public:
    float learningRate;
    ADAM(float learningRate_, int mbSize_, int maxEpochs_);
    void Optimize(NeuralNet* NN, Data* trainingData);
    void OptimizeInParallel(NeuralNet *NN, Data* trainingData);
    void OptimizeInParallel(NeuralNet *NN, Data* trainingData, Data* testData);
};

//class CUBIC: public Optimizer{
//public:
//    CUBIC(int mbSize_, int maxEpochs_);
//    void Optimize(NeuralNet* NN, Data* trainingData);
//};

#endif // __optimiser__
