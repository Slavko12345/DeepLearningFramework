#include "layers.h"
#include "realNumber.h"
#include "vect.h"
#include "matrix.h"
#include "tensor.h"
#include "tensor4D.h"
#include "globals.h"
#include "mathFunc.h"
#include "activityLayers.h"
#include "randomGenerator.h"
#include "activityData.h"
#include "architecture.h"
void layers::SetModel()
{
    Nlayers=5;

    layerList=new orderedData* [Nlayers];

    layerList[0] = new tensor(203, 32, 32);
    layerList[1] = new tensor(403, 16, 16);
    layerList[2] = new tensor(603, 8, 8);
    layerList[3] = new tensor(603, 1, 1);
    layerList[4] = new vect(10);
}

void layers::SetModel(architecture * arch){
    Nlayers = arch->Nlayers;
    layerList=new orderedData* [Nlayers];

    for(int j=0; j<Nlayers; ++j){
        if (arch->layer_dimension[j] == 1)
            layerList[j] = new vect(arch->layer_shape[j][0]);
        if (arch->layer_dimension[j] == 2)
            layerList[j] = new matrix(arch->layer_shape[j][0], arch->layer_shape[j][1]);
        if (arch->layer_dimension[j] == 3)
            layerList[j] = new tensor(arch->layer_shape[j][0], arch->layer_shape[j][1], arch->layer_shape[j][2]);
    }
}

void layers::SetInnerLayersToZero(){
    for(int l=1; l<Nlayers; l++)
        layerList[l]->SetToZero();
}

void layers::SetLayersToZero(){
    for(int l=0; l<Nlayers; l++)
        layerList[l]->SetToZero();
}

void layers::Print(){
    for(int l=0; l<Nlayers; l++)
        layerList[l]->Print();
}

void layers::SetInput(orderedData* input, activityLayers* actLayers, bool testMode){
    //in future can be changed to data augmentation function
    //needed for dropping units from input
    tensor* layer_0 = static_cast<tensor*>(layerList[0]);
    tensor* input_t = static_cast<tensor*>(input);
    const int numLayers = input_t->depth;
    tensor* input_layer_0 = new tensor();
    input_layer_0->SubTensor(layer_0, numLayers);

    if (!testMode && DATA_AUGMENTATION){
        input_layer_0->RandomlyTranform(input_t);
    }
    else
        input_layer_0->Copy(input_t);

    if (INPUT_NORMALIZATION){
        if (!LAYERWISE_NORMALIZATION){
            float mean, stDev;
            input_layer_0->NormalizeMeanStDev(input_layer_0, mean, stDev);
            if (APPEND_INPUT_STATISTICS){
                int linearLayerLen = layerList[Nlayers-2]->len;
                layerList[Nlayers-2]->elem[linearLayerLen - 2] = mean;
                layerList[Nlayers-2]->elem[linearLayerLen - 1] = stDev;
            }
        }
        else{
            int mean[numLayers], stDev[numLayers];
            float mean_, stDev_;
            matrix* input_j = new matrix();
            for(int j=0; j<numLayers; ++j){
                input_j->SetToTensorLayer(input_layer_0, j);
                input_j->NormalizeMeanStDev(input_j, mean_, stDev_);
                mean[j] = mean_;
                stDev[j] = stDev_;
            }
            DeleteOnlyShell(input_j);
            if (APPEND_INPUT_STATISTICS){
                int linearLayerLen = layerList[Nlayers-2]->len;
                for(int j=0; j<numLayers; ++j){
                    layerList[Nlayers-2]->elem[linearLayerLen - 2 * numLayers + 2 * j] = mean[j];
                    layerList[Nlayers-2]->elem[linearLayerLen - 2 * numLayers + 2 * j + 1] = stDev[j];
                }
            }
        }
    }

    if (!testMode)
        input_layer_0->SetDroppedElementsToZero(actLayers->layerList[0], input_layer_0->len);

    if (DROP_DATA_AUGMENTATION && !testMode){
        if (! randomGenerator::generateBool(INPUT_UNCHANGED_PROBABILITY)){
            actLayers->inputActivity->DropUnits();
            input_layer_0->SetDroppedElementsToZero(actLayers->inputActivity);
        }
    }

    layerList[0]->SetToZeroStartingFrom(input->len);

    DeleteOnlyShell(input_layer_0);
}

void layers::SetOutputDelta(layers* inputLayers, int trueClass){
    layerList[Nlayers-1]->Copy(inputLayers->layerList[Nlayers-1]);
    layerList[Nlayers-1]->elem[trueClass] -= 1.0;
    if (NEW_ERROR_FUNCTION)
        layerList[Nlayers-1]->Multiply(inputLayers->layerList[Nlayers-1]->elem[trueClass]);
}

layers::~layers(){
    for(int j=0; j<Nlayers; ++j)
        delete layerList[j];
    delete [] layerList;
}
