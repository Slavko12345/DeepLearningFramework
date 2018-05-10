#include "optimizer.h"
#include "neural_net.h"
#include "data.h"
#include "weights.h"
#include <omp.h>
#include "orderedData.h"
#include "matrix.h"
#include "globals.h"
#include "mathFunc.h"
#include <iostream>
#include <math.h>
#include "vect.h"
using namespace std;
Optimizer::Optimizer(int maxEpochs_, int mbSize_): maxEpochs(maxEpochs_), mbSize(mbSize_){
}

Optimizer::~Optimizer(){
}

void Optimizer::OptimizeInParallel(NeuralNet *NN, Data* trainingData){
    cout<<"Optimize in Parallel for the abstract class"<<endl;
}

void Optimizer::OptimizeInParallel(NeuralNet *NN, Data* trainingData, Data* testData){
    cout<<"Optimize in Parallel with test data for the abstract class"<<endl;
}


SGD::SGD(double learningRate_, int mbSize_, int maxEpochs_): Optimizer(maxEpochs_, mbSize_), learningRate(learningRate_){
}

void SGD::Optimize(NeuralNet* NN, Data* trainingData){
    int itersPerEpoch = ceil(trainingData->totalSize()/double(mbSize));
    double trainError, accuracy;

    Data * mbData = new Data;
    mbData->Initiate();

    for(int epoch=0; epoch<maxEpochs; ++epoch){
        for(int iter=0; iter<itersPerEpoch; ++iter){
            mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
            NN->CalculateGradient(mbData);
            NN->weightsData->Add(-learningRate,NN->gradient);
        }
        if (epoch % 5 == 0){
            NN->SwitchToTestMode();
            NN->CalculateErrorAndAccuracy(trainingData, trainError, accuracy);
            cout<<"epoch: "<<epoch<<" Error: "<<trainError<<" acc: "<<accuracy<<endl;
            NN->SwitchToTrainingMode();
            learningRate*=RATE_DECAY;
        }
    }
    DeleteOnlyDataShell(mbData);
}


RMSPROP::RMSPROP(double learningRate_, int mbSize_, int maxEpochs_): Optimizer(maxEpochs_, mbSize_), learningRate(learningRate_){
}

void RMSPROP::Optimize(NeuralNet* NN, Data* trainingData){
    int itersPerEpoch = ceil(trainingData->totalSize()/double(mbSize));

    double trainError, accuracy;


    Data * mbData = new Data;
    mbData->Initiate();

    weights* MS = new weights;
    MS->SetModel();
    MS->SetToZero();

    //first iteration
    NN->SwitchToTrainingMode();
    mbData->SelectMiniBatch(trainingData, 0, mbSize);
    NN->CalculateGradient(mbData);
    NN->weightsData->RmspropUpdate(NN->gradient, MS, 0.0, 1.0, learningRate);

    for(int epoch=0; epoch<maxEpochs; ++epoch){
        for(int iter=0; iter<itersPerEpoch; ++iter){
            mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
            NN->CalculateGradient(mbData);
            NN->weightsData->RmspropUpdate(NN->gradient, MS, 0.9, 0.1, learningRate);
        }
        if (epoch % 5 == 0){
            NN->SwitchToTestMode();
            NN->CalculateErrorAndAccuracy(trainingData, trainError, accuracy);
            cout<<"epoch: "<<epoch<<" Error: "<<trainError<<" l.rate: "<<learningRate<<" acc: "<<accuracy<<endl;
            NN->weightsData->WriteToFile((char*)NET_WEIGHTS_FILE);
            NN->SwitchToTrainingMode();
            learningRate*=RATE_DECAY;
        }
    }

    DeleteOnlyDataShell(mbData);
    delete MS;
}

ADAM::ADAM(double learningRate_, int mbSize_, int maxEpochs_): Optimizer(maxEpochs_, mbSize_), learningRate(learningRate_){
}

void ADAM::Optimize(NeuralNet* NN, Data* trainingData){
    int itersPerEpoch = ceil(trainingData->totalSize()/double(mbSize));
    double trainError, accuracy;

    Data * mbData = new Data;
    mbData->Initiate();

    weights* MS = new weights;
    MS->SetModel();
    MS->SetToZero();

    weights* Moment = new weights;
    Moment->SetModel();
    Moment->SetToZero();

    NN->SwitchToTrainingMode();

    double timeStart = omp_get_wtime();
    mbData->SelectMiniBatch(trainingData, 0, mbSize);
    NN->CalculateGradient(mbData);
    NN->weightsData->AdamUpdate(NN->gradient, Moment, MS, 0.0, 1.0, learningRate);

    for(int epoch=0; epoch<maxEpochs; ++epoch){
        for(int iter=0; iter<itersPerEpoch; ++iter){
            mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
            NN->CalculateGradient(mbData);
            NN->weightsData->AdamUpdate(NN->gradient, Moment, MS, 0.9, 0.1, learningRate);
        }
        if (epoch % 5 == 0){
            NN->SwitchToTestMode();
            NN->CalculateErrorAndAccuracy(trainingData, trainError, accuracy);
            cout<<"epoch: "<<epoch<<" Error: "<<trainError<<" l.rate: "<<learningRate<<" acc: "<<accuracy<<endl;
            NN->weightsData->WriteToFile((char*) NET_WEIGHTS_FILE);
            NN->SwitchToTrainingMode();
            learningRate*=RATE_DECAY;
        }
    }

    double timeEnd = omp_get_wtime();
    cout<<"Pure optimization time: "<<timeEnd - timeStart<<endl;

    DeleteOnlyDataShell(mbData);
    delete MS;
    delete Moment;
}


void ADAM::OptimizeInParallel(NeuralNet *NN, Data* trainingData){
    int numThreads;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }
    cout<<"Num Threads: "<<numThreads<<endl;

    int itersPerEpoch = ceil(trainingData->totalSize()/double(mbSize));
    double totalTrainError, accuracy, trainError[numThreads], maxAbsWeight, velocity[numThreads], time_Grad[numThreads];
    int correct[numThreads], totalCorrect;

    ofstream f(LOG_FILE);

    NeuralNet **NNList = new NeuralNet* [numThreads];
    NNList[0] = NN;
    NNList[0]->SwitchToTrainingMode();
    for(int j=1; j<numThreads; ++j){
        NNList[j] = new NeuralNet();
        NNList[j]->Initiate(NN);
        NNList[j]->SwitchToTrainingMode();
    }
    cout<<"Parallel nets are initiated"<<endl;

    Data* mbData = new Data;
    mbData->Initiate();

    Data** mbDataList = new Data* [numThreads];
    Data** trainingSubList = new Data* [numThreads];
    for(int j=0; j<numThreads; ++j){
        mbDataList[j] = new Data;
        mbDataList[j]->Initiate();
        trainingSubList[j] = new Data;
        trainingSubList[j]->Initiate();
    }
    cout<<"Parallel minibatches are initiated"<<endl;

    trainingData->SubDivide(trainingSubList, numThreads);
    cout<<"Training Data is subdivided"<<endl;

    weights* MS = new weights;
    MS->SetModel();
    MS->SetToZero();

    weights* Moment = new weights;
    Moment->SetModel();
    Moment->SetToZero();

    double timeStart = omp_get_wtime();

    //NN->SwitchToTrainingMode();

    mbData->SelectMiniBatch(trainingData, 0, mbSize);
    mbData->SubDivide(mbDataList, numThreads);

    #pragma omp parallel num_threads(numThreads)
    {
        //cout<<"Threads: "<<omp_get_num_threads()<<endl;
        double gradTimeStart = omp_get_wtime();
        int ID = omp_get_thread_num();
        //mbDataList[ID]->SelectMiniBatch(trainingData, mbSubSize[ID]);
        NNList[ID]->CalculateGradient(mbDataList[ID]);
        if (ADAPT_PROCESSORS_LOAD){
            time_Grad[ID] = omp_get_wtime() - gradTimeStart;
            velocity[ID] = (double) mbDataList[ID]->totalSize() / time_Grad[ID];
//            totalVelocity = 0;
//            for(int j=0; j<numThreads; ++j)
//                totalVelocity+=velocity[j];
//            for(int j=0; j<numThreads; ++j)
//                mbSubSize[j] = round(velocity[j] / totalVelocity * mbSize);
        }
    }

    for(int j=1; j<numThreads; ++j)
        NN->gradient->Add(NNList[j]->gradient);
    NN->weightsData->AdamUpdate(NN->gradient, Moment, MS, 0.0, 1.0, learningRate);
    double oldTime=omp_get_wtime(), newTime;

    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp single
        {
            cout<<"Main loop num threads: "<<omp_get_num_threads()<<endl;
            if (omp_get_num_threads() != numThreads)
                cout<<"ERROR: NUM THREADS CHANGED!!!"<<endl;
        }

        int ID = omp_get_thread_num(), correct_ID;
        double trainError_ID, gradTimeStart;

        for(int epoch=0; epoch<maxEpochs; ++epoch){
            for(int iter=0; iter<itersPerEpoch; ++iter){
                #pragma omp barrier
                #pragma omp single
                {
                    if (!ADAPT_PROCESSORS_LOAD){
                        mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
                        mbData->SubDivide(mbDataList, numThreads);
                    }
                    if (ADAPT_PROCESSORS_LOAD){
                        mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
                        mbData->SubDivide(mbDataList, numThreads, velocity);
                    }
                }

                gradTimeStart = omp_get_wtime();
                NNList[ID]->CalculateGradient(mbDataList[ID]);
                if (ADAPT_PROCESSORS_LOAD){
                    time_Grad[ID] = omp_get_wtime() - gradTimeStart;
                    velocity[ID] = 0.8 * velocity[ID] + 0.2 * (double) mbDataList[ID]->totalSize() / time_Grad[ID];
                }

                #pragma omp barrier
                #pragma omp single
                {
                    for(int j=1; j<numThreads; ++j)
                        NN->gradient->Add(NNList[j]->gradient);

                    NN->weightsData->AdamUpdate(NN->gradient, Moment, MS, 0.9, 0.1, learningRate);

                    if (NORMALIZE_WEIGHTS){
                        maxAbsWeight = NN->weightsData->MaxAbs();
                        if (maxAbsWeight > NORMALIZATION_RADIUS)
                            NN->weightsData->Multiply(NORMALIZATION_RADIUS / maxAbsWeight);
                    }

//                    if (ADAPT_PROCESSORS_LOAD){
//                        totalVelocity = 0;
//                        for(int j=0; j<numThreads; ++j)
//                            totalVelocity+=velocity[j];
//
//                        for(int j=0; j<numThreads; ++j)
//                            mbSubSize[j] = round(velocity[j] / totalVelocity * mbSize);
//                    }
                }
            }
            if ((epoch + 1) % 5 == 0){
//                #pragma omp barrier
//                #pragma omp single
//                {
//                    NN->SwitchToTestMode();
//                }
                #pragma omp barrier
                NNList[ID]->SwitchToTestMode();
                #pragma omp barrier

                NNList[ID]->CalculateSubErrorAndAccuracy(trainingSubList[ID], trainError_ID, correct_ID);
                correct[ID] = correct_ID;
                trainError[ID] = trainError_ID;
                #pragma omp barrier
                #pragma omp single
                {
                    totalCorrect = 0; totalTrainError = 0;
                    for(int j=0; j<numThreads; ++j){
                        totalCorrect += correct[j];
                        totalTrainError += trainError[j];
                    }
                    totalTrainError /= trainingData->totalSize();
                    accuracy = (double) totalCorrect / trainingData->totalSize();
                    maxAbsWeight = NN->weightsData->MaxAbs();
                    if (NEW_ERROR_FUNCTION)
                        cout<<"epoch: "<<epoch + 1<<" Error: "<<- totalTrainError<<" l.rate: "<<learningRate<<" acc: "<<accuracy<<" maxAbs: "<<maxAbsWeight<<endl;
                    else
                        cout<<"epoch: "<<epoch + 1<<" Error: "<<totalTrainError<<" l.rate: "<<learningRate<<" acc: "<<accuracy<<" maxAbs: "<<maxAbsWeight<<endl;

                    f<<"epoch: "<<epoch<<" Error: "<<totalTrainError<<" l.rate: "<<learningRate<<" acc: "<<accuracy<<" maxAbs: "<<maxAbsWeight<<endl;
                    for(int t=0; t<numThreads; ++t)
                        f<<velocity[t]<<" ";
                    f<<endl;
                    //NN->weightsData->WriteToFile((char*)NET_WEIGHTS_FILE);
                    learningRate *= RATE_DECAY;
                    //NN->SwitchToTrainingMode();
                    newTime = omp_get_wtime();
                    cout<<"Last 5 epochs time: "<<newTime - oldTime<<" finish in: "<<(maxEpochs - epoch) / 5 * (newTime - oldTime)<<" s"<<endl<<endl;
                    oldTime = newTime;
                }
                #pragma omp barrier
                NNList[ID]->SwitchToTrainingMode();
                #pragma omp barrier
            }
        }
    }

    double timeEnd = omp_get_wtime();
    cout<<"Pure optimization time: "<<timeEnd - timeStart<<endl;
    f<<"Pure optimization time: "<<timeEnd - timeStart<<endl;

    f.close();

    for(int j=0; j<numThreads; ++j){
        DeleteOnlyDataShell(mbDataList[j]);
        DeleteOnlyDataShell(trainingSubList[j]);
    }

    delete [] mbDataList;
    delete [] trainingSubList;

    NNList[0]=NULL;
    for(int j=1; j<numThreads; ++j)
        delete NNList[j];
    delete[] NNList;
    delete MS;
    delete Moment;
}





void ADAM::OptimizeInParallel(NeuralNet *NN, Data* trainingData, Data* testData){
    int numThreads;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }
    cout<<"Num Threads: "<<numThreads<<endl;

    int itersPerEpoch = ceil(trainingData->totalSize()/double(mbSize));
    double totalTrainError, totalTestError, accuracy, testAccuracy, trainError[numThreads], testError[numThreads], maxAbsWeight, velocity[numThreads], time_Grad[numThreads];
    int correct[numThreads], testCorrect[numThreads], totalCorrect, totalTestCorrect;

    ofstream f(LOG_FILE);

    NeuralNet **NNList = new NeuralNet* [numThreads];
    NNList[0] = NN;
    NNList[0]->SwitchToTrainingMode();
    for(int j=1; j<numThreads; ++j){
        NNList[j] = new NeuralNet();
        NNList[j]->Initiate(NN);
        NNList[j]->SwitchToTrainingMode();
    }
    cout<<"Parallel nets are initiated"<<endl;

    Data* mbData = new Data;
    mbData->Initiate();

    Data** mbDataList = new Data* [numThreads];
    Data** trainingSubList = new Data* [numThreads];
    Data** testSubList = new Data* [numThreads];
    for(int j=0; j<numThreads; ++j){
        mbDataList[j] = new Data;
        mbDataList[j]->Initiate();

        trainingSubList[j] = new Data;
        trainingSubList[j]->Initiate();

        testSubList[j] = new Data;
        testSubList[j]->Initiate();
    }
    cout<<"Parallel minibatches are initiated"<<endl;

    trainingData->SubDivide(trainingSubList, numThreads);
    testData->SubDivide(testSubList, numThreads);
    cout<<"Training Data is subdivided"<<endl;

    weights* MS = new weights;
    MS->SetModel();
    MS->SetToZero();

    weights* Moment = new weights;
    Moment->SetModel();
    Moment->SetToZero();

    double timeStart = omp_get_wtime();

    //NN->SwitchToTrainingMode();

    mbData->SelectMiniBatch(trainingData, 0, mbSize);
    mbData->SubDivide(mbDataList, numThreads);

    #pragma omp parallel num_threads(numThreads)
    {
        //cout<<"Threads: "<<omp_get_num_threads()<<endl;
        double gradTimeStart = omp_get_wtime();
        int ID = omp_get_thread_num();
        //mbDataList[ID]->SelectMiniBatch(trainingData, mbSubSize[ID]);
        NNList[ID]->CalculateGradient(mbDataList[ID]);
        if (ADAPT_PROCESSORS_LOAD){
            time_Grad[ID] = omp_get_wtime() - gradTimeStart;
            velocity[ID] = (double) mbDataList[ID]->totalSize() / time_Grad[ID];
//            totalVelocity = 0;
//            for(int j=0; j<numThreads; ++j)
//                totalVelocity+=velocity[j];
//            for(int j=0; j<numThreads; ++j)
//                mbSubSize[j] = round(velocity[j] / totalVelocity * mbSize);
        }
    }

    for(int j=1; j<numThreads; ++j)
        NN->gradient->Add(NNList[j]->gradient);
    NN->weightsData->AdamUpdate(NN->gradient, Moment, MS, 0.0, 1.0, learningRate);
    double oldTime=omp_get_wtime(), newTime;

    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp single
        {
            cout<<"Main loop num threads: "<<omp_get_num_threads()<<endl;
            if (omp_get_num_threads() != numThreads)
                cout<<"ERROR: NUM THREADS CHANGED!!!"<<endl;
        }

        int ID = omp_get_thread_num(), correct_ID, testCorrect_ID;
        double trainError_ID, testError_ID, gradTimeStart;

        for(int epoch=0; epoch<maxEpochs; ++epoch){
            for(int iter=0; iter<itersPerEpoch; ++iter){
                #pragma omp barrier
                #pragma omp single
                {
                    if (!ADAPT_PROCESSORS_LOAD){
                        mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
                        mbData->SubDivide(mbDataList, numThreads);
                    }
                    if (ADAPT_PROCESSORS_LOAD){
                        mbData->SelectMiniBatch(trainingData, iter*mbSize, mbSize);
                        mbData->SubDivide(mbDataList, numThreads, velocity);
                    }
                }

                gradTimeStart = omp_get_wtime();
                NNList[ID]->CalculateGradient(mbDataList[ID]);
                if (ADAPT_PROCESSORS_LOAD){
                    time_Grad[ID] = omp_get_wtime() - gradTimeStart;
                    velocity[ID] = 0.8 * velocity[ID] + 0.2 * (double) mbDataList[ID]->totalSize() / time_Grad[ID];
                }

                #pragma omp barrier
                #pragma omp single
                {
                    for(int j=1; j<numThreads; ++j)
                        NN->gradient->Add(NNList[j]->gradient);

                    NN->weightsData->AdamUpdate(NN->gradient, Moment, MS, 0.9, 0.1, learningRate);

                    if (NORMALIZE_WEIGHTS){
                        maxAbsWeight = NN->weightsData->MaxAbs();
                        if (maxAbsWeight > NORMALIZATION_RADIUS)
                            NN->weightsData->Multiply(NORMALIZATION_RADIUS / maxAbsWeight);
                    }

//                    if (ADAPT_PROCESSORS_LOAD){
//                        totalVelocity = 0;
//                        for(int j=0; j<numThreads; ++j)
//                            totalVelocity+=velocity[j];
//
//                        for(int j=0; j<numThreads; ++j)
//                            mbSubSize[j] = round(velocity[j] / totalVelocity * mbSize);
//                    }
                }
            }
            if ((epoch + 1) % 5 == 0){
//                #pragma omp barrier
//                #pragma omp single
//                {
//                    NN->SwitchToTestMode();
//                }
                #pragma omp barrier
                NNList[ID]->SwitchToTestMode();
                #pragma omp barrier

                NNList[ID]->CalculateSubErrorAndAccuracy(trainingSubList[ID], trainError_ID, correct_ID);
                correct[ID] = correct_ID;
                trainError[ID] = trainError_ID;


                NNList[ID]->CalculateSubErrorAndAccuracy(testSubList[ID], testError_ID, testCorrect_ID);
                testCorrect[ID] = testCorrect_ID;
                testError[ID] = testError_ID;


                #pragma omp barrier
                #pragma omp single
                {
                    totalCorrect = 0; totalTrainError = 0;
                    totalTestCorrect = 0; totalTestError = 0;
                    for(int j=0; j<numThreads; ++j){
                        totalCorrect += correct[j];
                        totalTrainError += trainError[j];

                        totalTestCorrect += testCorrect[j];
                        totalTestError += testError[j];
                    }
                    totalTrainError /= trainingData->totalSize();
                    accuracy = (double) totalCorrect / trainingData->totalSize();

                    totalTestError /= testData->totalSize();
                    testAccuracy = (double) totalTestCorrect / testData->totalSize();

                    maxAbsWeight = NN->weightsData->MaxAbs();
                    if (NEW_ERROR_FUNCTION)
                        cout<<"epoch: "<<epoch + 1<<" Tr. Err: "<<- totalTrainError<<" l.rate: "<<learningRate<<" acc: "<<accuracy<<" maxAbs: "<<maxAbsWeight<<endl;
                    else
                        cout<<"epoch: "<<epoch + 1<<" Tr. Err: "<<totalTrainError<<" Test E: "<<totalTestError<<
                        " l.rate: "<<learningRate<<" Tr. acc: "<<accuracy<<" Test acc: "<< testAccuracy <<" maxAbs: "<<maxAbsWeight<<endl;

                    f<<"epoch: "<<epoch<<"Train Error: "<<totalTrainError<<" Test E: "<<totalTestError<<
                    " l.rate: "<<learningRate<<" Train acc: "<<accuracy<<" Test acc: "<< testAccuracy <<" maxAbs: "<<maxAbsWeight<<endl;
                    //for(int t=0; t<numThreads; ++t)
                    //    f<<velocity[t]<<" ";
                    //f<<endl;
                    //NN->weightsData->WriteToFile((char*)NET_WEIGHTS_FILE);
                    learningRate *= RATE_DECAY;
                    //NN->SwitchToTrainingMode();
                    newTime = omp_get_wtime();
                    cout<<"Last 5 epochs time: "<<newTime - oldTime<<" finish in: "<<(maxEpochs - epoch) / 5 * (newTime - oldTime)<<" s"<<endl<<endl;
                    oldTime = newTime;
                }
                #pragma omp barrier
                NNList[ID]->SwitchToTrainingMode();
                #pragma omp barrier
            }
        }
    }

    double timeEnd = omp_get_wtime();
    cout<<"Pure optimization time: "<<timeEnd - timeStart<<endl;
    f<<"Pure optimization time: "<<timeEnd - timeStart<<endl;

    f.close();

    for(int j=0; j<numThreads; ++j){
        DeleteOnlyDataShell(mbDataList[j]);
        DeleteOnlyDataShell(trainingSubList[j]);
    }

    delete [] mbDataList;
    delete [] trainingSubList;

    NNList[0]=NULL;
    for(int j=1; j<numThreads; ++j)
        delete NNList[j];
    delete[] NNList;
    delete MS;
    delete Moment;
}






//
//CUBIC::CUBIC(int mbSize_, int maxIterations_): Optimizer(maxIterations_, mbSize_){
//}
//
//void CUBIC::Optimize(NeuralNet* NN, Data* trainingData){
//    int itersPerEpoch = trainingData->totalSize/(trainingData->C * mbSize);
//    maxIterations = itersPerEpoch * maxEpochs;
//    double trainError, accuracy;
//    double epsHessian = 0.1;
//    Data * mbData = new Data;
//    mbData->AllocateMemory(trainingData->C, mbSize, mbSize);
//
//    weights* grad = new weights;
//    grad->SetModel();
//
//    weights* d1 = new weights;
//    d1->SetModel();
//
//    weights* d2 = new weights;
//    d2->SetModel();
//
//    weights* wCopy = new weights();
//    wCopy->SetModel();
//
//    weights* moment = new weights;
//    moment->SetModel();
//    moment->SetToZero();
//
//    weights* Hd1 = new weights();
//    Hd1->SetModel();
//
//    weights* Hd2 = new weights;
//    Hd2->SetModel();
//
//    //First Step
//    //NN->SetRandomWeights(0.01);
//    mbData->SelectRandomMiniBatch(trainingData, mbSize);
//    NN->CalculateGradient(mbData);
//    NN->weightsData->Add(-1E-5,NN->gradient);
//    moment->Add(-1E-5,NN->gradient);
//
//    matrix* B = new matrix(2,2);
//    vect* r = new vect(2);
//    vect* alpha = new vect(2);
//    vect* rV = new vect(2);
//
//    vect* eigenValues = new vect(2);
//    matrix* eigenVectors = new matrix(2, 2);
//
//    double g_g;
//    double trust_region_size = 1.0;
//    double eps;
//    double f0, f1, f2;
//    double new_f, new_acc;
//    for(int iter=0; iter<maxIterations; iter++){
//        mbData->SelectRandomMiniBatch(trainingData, mbSize);
//
//
//        NN->CalculateGradientFunctionValue(mbData, f0);
//        grad->Copy(NN->gradient);
//        FormOrthonormalBasis(grad, moment, d1, d2, g_g);
//
//        wCopy->Copy(NN->weightsData);
//
//        NN->weightsData->Add(epsHessian, d1);
//        NN->CalculateGradientFunctionValue(mbData, f1);
//        Hd1->SetToLinearCombination(1.0 / epsHessian, - 1.0 / epsHessian, NN->gradient, grad);
//        NN->weightsData->Copy(wCopy);
//
//        NN->weightsData->Add(epsHessian, d2);
//        NN->CalculateGradientFunctionValue(mbData, f2);
//        Hd2->SetToLinearCombination(1.0 / epsHessian, - 1.0 / epsHessian, NN->gradient, grad);
//        NN->weightsData->Copy(wCopy);
//
//        FormHessianInSubspace(d1, d2, Hd1, Hd2, B);
//        r->elem[0] = g_g; r->elem[1] = 0;
//        B->EigenDecompose(eigenValues, eigenVectors);
//        rV->TrMatrProd(eigenVectors, r);
//
//        eps = trust_region_size;
//        while(1){
//            alpha->FindTrustRegionMinima(eigenValues, eigenVectors, rV, eps);
//            NN->weightsData->Add(-alpha->elem[0], d1);
//            NN->weightsData->Add(-alpha->elem[1], d2);
//            NN->CalculateErrorAndAccuracy(mbData, new_f, new_acc);
//            cout<<"eps: "<<eps<<" oldF: "<<f0<<" newF: "<<new_f<<endl;
//            if (new_f < f0) break;
//            eps /= 2.0;
//            NN->weightsData->Copy(wCopy);
//        }
//
//
//
//
//
//
//
//
//
//        //NN->weightsData->Add(-learningRate,NN->gradient);
//
//
//
//
//        if (iter%(itersPerEpoch)==0){
//            NN->CalculateErrorAndAccuracy(trainingData, trainError, accuracy);
//            cout<<"iter: "<<iter<<" Error: "<<trainError<<" accuracy: "<<accuracy<<endl;
//        }
//    }
//
//    DeleteOnlyDataShell(mbData);
//}
