#ifndef __layers__
#define __layers__

class orderedData;
class activityData;

class layers{
public:
    int Nlayers;
    orderedData ** layerList;
    void SetModel();
    void SetInnerLayersToZero();
    void SetLayersToZero();
    void Print();
    void SetInput(orderedData * input, activityData* inputActivity, bool testMode);
    void SetOutputDelta(layers* inputLayers, int trueClass);
    ~layers();
};

#endif // __layers__
