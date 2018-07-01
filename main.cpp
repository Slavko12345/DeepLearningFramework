#include <iostream>
#include "computationalNode.h"
#include "vect.h"
#include "optimizer.h"
#include "data.h"
#include "neural_net.h"
#include "matrix.h"
#include <math.h>
#include "weights.h"
#include "tensor.h"
#include <stdlib.h>
#include "activityData.h"
#include "mathFunc.h"
#include "randomGenerator.h"
#include "realNumber.h"
//#include <windows.h>
#include "globals.h"
#include <xmmintrin.h>
#include <omp.h>
#include <cstdlib>
#include "tensor4D.h"
#include <algorithm>
#include "orderedData.h"
#include "vect.h"
#include "computationalModel.h"
#include "architecture.h"

using namespace std;

string GetEnv( const string & var ) {
     const char * val = ::getenv( var.c_str() );
     if ( val == 0 ) {
         return "";
     }
     else {
         return val;
     }
}

void SomeInitialStuff(){
    randomGenerator::SetRandomSeed();
    //float time_start = omp_get_wtime();
//    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
//    //putenv((char*)"OMP_PROC_BIND=TRUE");
//    putenv((char*)"OMP_DYNAMIC=FALSE");
    //putenv((char*)"OMP_WAIT_POLICY=ACTIVE");
}

int main()
{
    SomeInitialStuff();

    Data * CifarTrain = new Data, * CifarTest = new Data;

    CifarTrain->ReadTrainingCifar10();
    CifarTest->ReadTestCifar10();

    NeuralNet *NN = new NeuralNet;

    const int num_layers = 2;
    const int layer_size = 10;

    NN->SetInputShape(3, 32, 32);

    NN->Add(StairsFullBottleneck::Start(num_layers, layer_size, layer_size));
    NN->Add(FullColumnDrop::Start(2, 2, 2 * layer_size, 2));

    NN->Add(StairsFullBottleneckBalancedDrop::Start(num_layers, layer_size, layer_size));
    NN->Add(FullColumnDrop::Start(2, 2, 2 * layer_size, 2));

    NN->Add(StairsFullBottleneckBalancedDrop::Start(num_layers, layer_size, layer_size));

    NN->Add(FullColumnDrop::Start(8, 8, 2 * layer_size, 4));

    NN->Add(FullyConnectedSoftMax::Start(10));
    NN->Initiate();

    cout<<"Net is initiated"<<endl;

    Optimizer* Opt = new ADAM(LEARNING_RATE, MINIBATCH_SIZE, NUM_EPOCHS);
    Opt->OptimizeInParallel(NN, CifarTrain, CifarTest);
    NN->weightsData->WriteToFile((char*)NET_WEIGHTS_FILE);
    cout<<"Optimization is done"<<endl;

    float error, accuracy;
    NN->CalculateErrorAndAccuracy(CifarTest, error, accuracy);
    cout<<"Test error: "<<error<<"; Test accuracy: "<<accuracy*100.0f<<"%"<<endl;

    ofstream f("data/result.txt");
    f<<"CIFAR-10 test set classification: "<<endl;
    f<<"error: "<<error<<" accuracy: "<<accuracy*100.0f<<"%"<<endl;
    f.close();

    delete CifarTrain;
    delete CifarTest;
    delete NN;
    delete Opt;

    return 0;
}
