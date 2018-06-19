#ifndef __computationalModel__
#define __computationalModel__
#include <vector>
#include <tuple>
using namespace std;

struct computationalNode;
struct layers;
struct weights;
struct activityLayers;
class architecture;


struct computationalModel{
    int Nlayers;

    vector<tuple<int, int, computationalNode *> > computationList;
    //vector<pair<int, int> > computationList;
    vector<bool> hasBottomWeightDependency;
    void AddNode(int from, int to, computationalNode* node);
    void SetModel(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, bool primalWeightOwner);
    void SetModel(layers* layersData, layers* deltas, weights* weightsData, weights* gradient,
                  activityLayers* layersActivity, bool primalWeightOwner, computationalModel * primalCM);

    void SetModel(architecture * arch, layers* layersData, layers* deltas, weights* weightsData, weights* gradient,
                  activityLayers* layersActivity, bool primalWeightOwner);

    void SetModel(architecture * arch, layers* layersData, layers* deltas, weights* weightsData, weights* gradient,
                  activityLayers* layersActivity, bool primalWeightOwner, computationalModel * primalCM);

    void ForwardPass();
    void BackwardPass(int trueClass);
    void SetToTrainingMode();
    void SetToTestMode();
    void UpdateBalancedDropParameters(float alpha_, float pDrop_, float pNotDrop_);
    void Compile(layers* layersData, layers* deltas, weights* weightsData, weights* gradient, activityLayers* layersActivity, bool primalWeightOwner);
    void WriteCoefficientsToFile();
    ~computationalModel();
};

#endif // __computationalModel__
