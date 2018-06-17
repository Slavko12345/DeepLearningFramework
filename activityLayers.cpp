#include "activityLayers.h"
#include "activityData.h"
#include "layers.h"
#include "orderedData.h"
#include "vect.h"
#include "matrix.h"
#include "tensor.h"
#include "globals.h"
activityLayers::activityLayers(){
}

void activityLayers::SetModel(layers* layersData){
    Nlayers = layersData->Nlayers;
    dropping = false;

    layerList = new activityData* [Nlayers];
    dropoutRates.resize(Nlayers, 0.0f);

//    dropoutRates[0] = 0.0625;
//    dropoutRates[1] = 0.125;
//    dropoutRates[2] = 0.125;
//    dropoutRates[3] = 0.125;
//    dropoutRates[4] = 0.125;


    orderedData* l_j;
    for(int j=0; j<Nlayers; j++){
        l_j = layersData->layerList[j];
        layerList[j] = new activityData(l_j->len, dropoutRates[j]);
        if (layerList[j]->dropping) dropping = true;
    }

    inputActivity = new activityData(INPUT_LEN, INPUT_DROP_RATE * DROP_DATA_AUGMENTATION);
}

void activityLayers::SetAllActive(){
    for(int j=0; j<Nlayers; j++)
        layerList[j]->SetAllActive();
}

void activityLayers::DropUnits(){
    for(int j=0; j<Nlayers; ++j)
        layerList[j]->DropUnits();
}

activityLayers::~activityLayers(){
    for(int j=0; j<Nlayers; ++j)
        delete layerList[j];
    delete [] layerList;
}


