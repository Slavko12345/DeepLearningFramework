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
//    putenv((char*)"OMP_WAIT_POLICY=ACTIVE");
}

int main()
{
    SomeInitialStuff();

    Data * CifarTrain = new Data, * CifarTest = new Data;

    CifarTrain->ReadTrainingCifar10();
    CifarTest->ReadTestCifar10();

    NeuralNet *NN = new NeuralNet;

    NN->SetInputShape(3, 32, 32);

    NN->Add(StairsFullBottleneckBalancedDrop::Start(10, 10, 10));
    NN->Add(FullAveragePoolingBalancedDrop::Start());

    NN->Add(StairsFullBottleneckBalancedDrop::Start(10, 10, 10));
    NN->Add(FullAveragePoolingBalancedDrop::Start());

    NN->Add(StairsFullBottleneckBalancedDrop::Start(10, 10, 10));

    NN->Add(FullAveragePoolingBalancedDrop::Start(8, 8));

    NN->Add(FullyConnectedSoftMax::Start(10));
    NN->Initiate();

    cout<<"Net is initiated"<<endl;

    Optimizer* Opt = new ADAM(LEARNING_RATE, MINIBATCH_SIZE, NUM_EPOCHS);
    Opt->OptimizeInParallel(NN, CifarTrain, CifarTest);
    NN->weightsData->WriteToFile((char*)NET_WEIGHTS_FILE);
    cout<<"Optimization is done"<<endl;

    NN->SwitchToTestMode();

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
