#ifndef __activityLayers__
#define __activityLayers__
#include <vector>
using namespace std;

class layers;
class activityData;

struct activityLayers{
    int Nlayers;
    activityData **layerList;
    vector<double> dropoutRates;
    bool dropping;
    activityData *inputActivity;

    activityLayers();
    void SetModel(layers* layersData);
    void SetAllActive();
    void DropUnits();
    ~activityLayers();
};

#endif // __activityLayers__
