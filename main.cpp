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



int main()
{

    randomGenerator::SetRandomSeed();
    float time_start = omp_get_wtime(), error, accuracy;
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    //putenv((char*)"OMP_PROC_BIND=TRUE");
    putenv((char*)"OMP_DYNAMIC=FALSE");
    putenv((char*)"OMP_WAIT_POLICY=ACTIVE");
    omp_set_num_threads(MAX_THREADS);

    Data* CifarTrain = new Data;
    CifarTrain->ReadTrainingCifar10();

    Data*  CifarTest = new Data;
    CifarTest->ReadTestCifar10();

    NeuralNet *NN=new NeuralNet;
    NN->Initiate();
    //NN->weightsData->ReadFromFile((char*)NET_WEIGHTS_FILE);
    cout<<"Net is initiated"<<endl;
    cout<<"Number of coefficients: "<<NN->weightsData->GetWeightLen()<<endl;

    Optimizer* Opt = new ADAM(LEARNING_RATE, MINIBATCH_SIZE, NUM_EPOCHS);
    //Opt->Optimize(NN, CifarTrain);
    Opt->OptimizeInParallel(NN, CifarTrain, CifarTest);
    NN->weightsData->WriteToFile((char*)NET_WEIGHTS_FILE);
    //Opt->Optimize(NN, Cifar10Train);
    cout<<"Optimization is done"<<endl;

    NN->SwitchToTestMode();

    NN->CalculateErrorAndAccuracy(CifarTest, error, accuracy);
    cout<<"error: "<<error<<" accuracy: "<<accuracy*100.0f<<"%"<<endl;

    ofstream f("data/result.txt");
    f<<"CIFAR-10 test set classification: "<<endl;
    f<<"error: "<<error<<" accuracy: "<<accuracy*100.0f<<"%"<<endl;
    f.close();

    NN->computation->WriteCoefficientsToFile();

    delete CifarTrain;
    delete CifarTest;
    delete NN;
    delete Opt;

    cout<<"Execution time: "<<omp_get_wtime() - time_start<<endl;
    return 0;
}
