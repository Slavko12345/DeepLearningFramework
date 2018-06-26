#ifndef __neural_net__
#define __neural_net__

class computationalModel;
class Data;
class layers;
class weights;
class orderedData;
class activityLayers;
class architecture;
class computationalNode;

class NeuralNet
{
    public:
        int Nlayers;
        int Nweights;

        bool testMode;    //0: training mode; 1 - testMode;
        bool primalWeightOwner;

        bool architectureBased = false;

        architecture * arch;

        layers*  layersData;
        layers*  deltas;

        weights* weightsData;
        weights* gradient;

        activityLayers* layersActivity;

        computationalModel* computation;

        NeuralNet();

        void SetInputShape(int dim1, int dim2 = -1, int dim3 = -1);
        void Add(computationalNode * node);

        void ForwardPass(orderedData* input);
        void BackwardPass(int trueClass, int input_len);
        void ForwardBackwardPass(orderedData* input, int trueClass);

        void Initiate();
        void Initiate(NeuralNet * NN);

        void InitiateFromArchitecture();
        void InitiateFromArchitecture(NeuralNet * NN);

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

        bool CheckCompatibility(Data * data);

        ~NeuralNet();
};

#endif // __neural_net__
