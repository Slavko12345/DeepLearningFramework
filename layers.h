#ifndef __layers__
#define __layers__

class orderedData;
class activityLayers;
class architecture;

class layers{
public:
    int Nlayers;
    orderedData ** layerList;
    void SetModel();
    void SetModel(architecture * arch);
    void SetInnerLayersToZero();
    void SetLayersToZero();
    void Print();
    void SetInput(orderedData * input, activityLayers* actLayers, bool testMode);
    void SetOutputDelta(layers* inputLayers, int trueClass);
    ~layers();
};

#endif // __layers__
