#ifndef __neural_net__
#define __neural_net__

class computationalModel;
class Data;
class layers;
class weights;
class orderedData;
class activityLayers;

class NeuralNet
{
    public:
        int Nlayers;
        int Nweights;

        bool testMode;    //0: training mode; 1 - testMode;
        bool primalWeightOwner;

        layers*  layersData;
        layers*  deltas;

        weights* weightsData;
        weights* gradient;

        activityLayers* layersActivity;

        computationalModel* computation;

        void ForwardPass(orderedData* input);
        void BackwardPass(int trueClass, int input_len);
        void ForwardBackwardPass(orderedData* input, int trueClass);

        void Initiate();
        void Initiate(NeuralNet * NN);
        void SetRandomWeights(double bound);
        void CalculateGradient(Data* inputData);
        void CalculateGradientFunctionValue(Data* inputData, double& functionVal);
        void CalculateErrorAndAccuracy(Data* inputData, double &error, double &accuracy);
        void CalculateSubErrorAndAccuracy(Data* inputData, double &error, int &correct);
        double CalculateAccuracy(Data* inputData);
        void PrintProbabilities();
        void SwitchToTrainingMode();
        void SwitchToTestMode();
        void UpdateBalancedDropParameters(double alpha_, double pDrop_, double pNotDrop_);
        ~NeuralNet();
};

#endif // __neural_net__
