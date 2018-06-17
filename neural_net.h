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
        void SetRandomWeights(float bound);
        void CalculateGradient(Data* inputData);
        void CalculateGradientFunctionValue(Data* inputData, float& functionVal);
        void CalculateErrorAndAccuracy(Data* inputData, float &error, float &accuracy);
        void CalculateSubErrorAndAccuracy(Data* inputData, float &error, int &correct);
        float CalculateAccuracy(Data* inputData);
        void PrintProbabilities();
        void SwitchToTrainingMode();
        void SwitchToTestMode();
        void UpdateBalancedDropParameters(float alpha_, float pDrop_, float pNotDrop_);
        ~NeuralNet();
};

#endif // __neural_net__
