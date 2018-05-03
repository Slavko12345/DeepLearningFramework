#include "mathFunc.h"

#include "realNumber.h"
#include "vect.h"
#include "matrix.h"
#include "tensor.h"
#include "tensor4D.h"
#include <iostream>
#include <math.h>
#include "activityData.h"
#include "randomGenerator.h"
#include <algorithm>
#include "globals.h"
#include "matrix.h"
using namespace std;

double Sigma::f(const double & x){
    if (x>0) return 1.0/(1.0+exp(-x));
    return exp(x)/(1.0+exp(x));
}

double Sigma::df(const double & x){
    double s=f(x);
    return s*(1.0-s);
}

double Sigma::ddf(const double & x){
    double s=f(x);
    return s*(1.0-s)*(1.0-2.0*s);
}



double Tanh::f(const double & x){
    return tanh(x);
}

double Tanh::df(const double & x){
    double t=tanh(x);
    return 1.0-t*t;
}


double Tanh::ddf(const double & x){
    double t=tanh(x);
    return -2.0*t*(1-t*t);
}



double Relu::f(const double & x){
    return (x>0)*x;
}

double Relu::df(const double & x){
    return (x>0);
}

double Relu::ddf(const double & x){
    return 0;
}



void SoftMax(orderedData* inp, orderedData* out)
{
    double maxIn=inp->Max(), sumExp=0;
    double* inp_elem = inp->elem;
    double *out_elem = out->elem;
    for(int j=0; j<inp->len; ++j)
    {
        out_elem[j] = exp(inp_elem[j]-maxIn);
        sumExp += out_elem[j];
    }
    for(int j=0; j<inp->len; ++j)
        out_elem[j]/=sumExp;
}


void DeleteOnlyShell(orderedData* link){
    link->elem=NULL;
    delete link;
}

void DeleteOnlyShellActivity(activityData* link){
    link->activeUnits=NULL;
    delete link;
}

double TrustRegionFunc(double lamb, vect* eigenValues, vect* rV, double eps){
    double res = 0;
    for(int j=0; j<eigenValues->len; ++j)
        res += sqr(rV->elem[j] / (eigenValues->elem[j] + lamb));
    return res - sqr(eps);
}

double TrustRegionFuncDeriv(double lamb, vect* eigenValues, vect* rV, double eps){
    double res = 0;
    for(int j=0; j<eigenValues->len; ++j){
        //cout<<lamb<<endl;
        res -= sqr(rV->elem[j] / (eigenValues->elem[j] + lamb)) / (eigenValues->elem[j] + lamb);
    }
    return res * 2.0;
}



void Convolute2D2D(matrix* input, matrix* output, matrix* reversedKernel, realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow){

    double * input_iR, *output_oR, *revKernel_iR__oR, *output_elem;
    double   input_iR_iC;
    const double eps = 1E-10;
    int tempInd, indexInputSize, minOutRow, maxOutRow, minOutCol, maxOutCol;
    int kR = reversedKernel->rows, kC = reversedKernel->cols;
    int input_rows = input->rows, input_cols = input->cols;
    int maxR = - kR + 1 + paddingR, maxC = - kC + 1 + paddingC;
    int minR = input_rows - kR + 2 * paddingR, minC = input_cols - kC + 2*paddingC;
    int tempC = kC - paddingC - 1;
    int temp_iR;

    output_elem = output->elem;

    if (bias){
        double bias_ =  bias->elem[0];
        for(int j=0; j<output->len; ++j)
            output_elem[j] += bias_;
    }

    for(int iR=0; iR<input->rows; ++iR){
        input_iR = input->Row(iR);
        //build the map of input activity units
        indexInputRow.resize(0);
        for(int iC=0; iC<input->cols; ++iC)
            if (fabs(input_iR[iC])>eps)
                indexInputRow.push_back(iC);

        indexInputSize = indexInputRow.size();

        minOutRow = max(0, iR + maxR);
        maxOutRow = min(iR+paddingR, minR);
        temp_iR = maxR + iR;

        for(int oR=minOutRow; oR<=maxOutRow; ++oR){
            output_oR = output->Row(oR);
            revKernel_iR__oR = reversedKernel->Row(oR - temp_iR);

            for(int iCIndex=0, iC; iCIndex<indexInputSize; ++iCIndex){
                iC = indexInputRow[iCIndex];
                minOutCol = max(0, iC + maxC);
                maxOutCol = min(iC + paddingC, minC);
                input_iR_iC = input_iR[iC];
                tempInd = tempC - iC;
                for(int oC=minOutCol; oC<=maxOutCol; ++oC){
                    output_oR[oC]+=input_iR_iC * revKernel_iR__oR[tempInd + oC];//    kernel_iR__oR[iC-oC];
                }
            }
        }
    }
}

//
//void ForwardConvoluteStandardByIndex(double* input, double* output, const int siz, double* revKernel, const vector<int> & indexInputRow){
//    const int indexInputSize = indexInputRow.size();
//    double input_c = input[0];
//    output[0] += input_c * revKernel[1];
//    output[1] += input_c * revKernel[2];
//    for(int iCIndex=0, iC; iCIndex<indexInputSize; ++iCIndex){
//        iC = indexInputRow[iCIndex];
//        input_c = input[iC];
//        output[iC-1] += input_c * revKernel[0];
//        output[iC]   += input_c * revKernel[1];
//        output[iC+1] += input_c * revKernel[2];
//    }
//    input_c = input[siz - 1];
//    output[siz-2] += input_c * revKernel[0];
//    output[siz-1] += input_c * revKernel[1];
//}
//
//
//void ForwardConvoluteStandardCreateIndex(double* input, double* output, const int siz, double* revKernel, vector<int> & indexInputRow){
//    indexInputRow.resize(0);
//    const double eps = 1E-10;
//    double input_c = input[0];
//    output[0] += input_c * revKernel[1];
//    output[1] += input_c * revKernel[2];
//    for(int iC=1; iC<siz-1; ++iC){
//        if (fabs(input[iC])>eps){
//            indexInputRow.push_back(iC);
//            input_c = input[iC];
//            output[iC-1] += input_c * revKernel[0];
//            output[iC]   += input_c * revKernel[1];
//            output[iC+1] += input_c * revKernel[2];
//        }
//    }
//    input_c = input[siz - 1];
//    output[siz-2] += input_c * revKernel[0];
//    output[siz-1] += input_c * revKernel[1];
//}

void ConvoluteStandard(double* input, double* output, const int siz, double* kernel){
    output[0]+=input[0] * kernel[1] + input[1] * kernel[2];
    for(int oC=1; oC<siz-1; ++oC)
        output[oC] += input[oC-1] * kernel[0] + input[oC] * kernel[1] + input[oC+1] * kernel[2];
    output[siz-1] += input[siz-2] * kernel[0] + input[siz-1] * kernel[1];
}


void ConvoluteStandardSymmetric(double* input, double* output, const int siz, double* kernel){
    output[0]+=input[0] * kernel[1] + input[1] * kernel[0];
    for(int oC=1; oC<siz-1; ++oC)
        output[oC] += (input[oC-1] + input[oC+1]) * kernel[0] + input[oC] * kernel[1];
    output[siz-1] += input[siz-2] * kernel[0] + input[siz-1] * kernel[1];
}

void ConvoluteSymmetricNoPadding(double* input, double* output, const int siz, double* kernel){
    for(int oC=1; oC<siz-1; ++oC)
        output[oC] += (input[oC-1] + input[oC+1]) * kernel[0] + input[oC] * kernel[1];
}

void ConvoluteStandardFullySymmetric0(double* input, double* output, const int siz, double* kernel){
    output[0]+=(input[0] + input[1]) * kernel[0];
    for(int oC=1; oC<siz-1; ++oC)
        output[oC] += (input[oC-1] + input[oC] + input[oC+1]) * kernel[0];
    output[siz-1] += (input[siz-2] + input[siz-1]) * kernel[0];
}





////paddingR = paddingC=1; kernel->rows = kernel->cols = 3
//void ConvoluteStandard2D2D(matrix* input, matrix* output, matrix* reversedKernel, realNumber* bias, vector<int> & indexInputRow){
//    if (bias)
//        output->Add(bias->elem[0]);
//    int input_rows = input->rows, input_cols = input->cols;
//
//    double* input_iR = input->Row(0);
//    ForwardConvoluteStandardCreateIndex (input_iR, output->Row(0), input_cols, reversedKernel->Row(1), indexInputRow);
//    ForwardConvoluteStandardByIndex     (input_iR, output->Row(1), input_cols, reversedKernel->Row(2), indexInputRow);
//
//    for(int iR=1; iR<input_rows-1; ++iR){
//        input_iR = input->Row(iR);
//        ForwardConvoluteStandardCreateIndex (input_iR, output->Row(iR-1), input_cols, reversedKernel->Row(0), indexInputRow);
//        ForwardConvoluteStandardByIndex     (input_iR, output->Row(iR),   input_cols, reversedKernel->Row(1), indexInputRow);
//        ForwardConvoluteStandardByIndex     (input_iR, output->Row(iR+1), input_cols, reversedKernel->Row(2), indexInputRow);
//    }
//
//    input_iR = input->Row(input_rows-1);
//    ForwardConvoluteStandardCreateIndex (input_iR, output->Row(input_rows-2), input_cols, reversedKernel->Row(0), indexInputRow);
//    ForwardConvoluteStandardByIndex     (input_iR, output->Row(input_rows-1), input_cols, reversedKernel->Row(1), indexInputRow);
//}


void ConvoluteStandard2D2D(matrix* input, matrix* output, matrix* kernel){
//    if (bias)
//        output->Add(bias->elem[0]);

    int input_rows = input->rows, input_cols = input->cols;

    double* output_oR = output->Row(0);
    ConvoluteStandard(input->Row(0), output_oR, input_cols, kernel->Row(1));
    ConvoluteStandard(input->Row(1), output_oR, input_cols, kernel->Row(2));

    for(int oR=1; oR<input_rows-1; ++oR){
        output_oR = output->Row(oR);
        ConvoluteStandard(input->Row(oR-1), output_oR, input_cols, kernel->Row(0));
        ConvoluteStandard(input->Row(oR),   output_oR, input_cols, kernel->Row(1));
        ConvoluteStandard(input->Row(oR+1), output_oR, input_cols, kernel->Row(2));
    }

    output_oR = output->Row(input_rows-1);
    ConvoluteStandard(input->Row(input_rows-2), output_oR, input_cols, kernel->Row(0));
    ConvoluteStandard(input->Row(input_rows-1), output_oR, input_cols, kernel->Row(1));
}

void ConvoluteStandard2D2DSymmetric(matrix* input, matrix* output, matrix* kernel){
    int input_rows = input->rows, input_cols = input->cols;

    double* output_oR = output->Row(0);
    ConvoluteStandardSymmetric(input->Row(0), output_oR, input_cols, kernel->Row(1));
    ConvoluteStandardSymmetric(input->Row(1), output_oR, input_cols, kernel->Row(2));

    for(int oR=1; oR<input_rows-1; ++oR){
        output_oR = output->Row(oR);
        ConvoluteStandardSymmetric(input->Row(oR-1), output_oR, input_cols, kernel->Row(0));
        ConvoluteStandardSymmetric(input->Row(oR),   output_oR, input_cols, kernel->Row(1));
        ConvoluteStandardSymmetric(input->Row(oR+1), output_oR, input_cols, kernel->Row(2));
    }

    output_oR = output->Row(input_rows-1);
    ConvoluteStandardSymmetric(input->Row(input_rows-2), output_oR, input_cols, kernel->Row(0));
    ConvoluteStandardSymmetric(input->Row(input_rows-1), output_oR, input_cols, kernel->Row(1));
}

void ConvoluteStandard2D2DSymmetric2(matrix* input, matrix* output, matrix* kernel){
    int input_rows = input->rows, input_cols = input->cols;

    double* output_oR = output->Row(0);
    ConvoluteStandardSymmetric(input->Row(0), output_oR, input_cols, kernel->Row(1));
    ConvoluteStandardSymmetric(input->Row(1), output_oR, input_cols, kernel->Row(0));

    for(int oR=1; oR<input_rows-1; ++oR){
        output_oR = output->Row(oR);
        ConvoluteStandardSymmetric(input->Row(oR-1), output_oR, input_cols, kernel->Row(0));
        ConvoluteStandardSymmetric(input->Row(oR),   output_oR, input_cols, kernel->Row(1));
        ConvoluteStandardSymmetric(input->Row(oR+1), output_oR, input_cols, kernel->Row(0));
    }

    output_oR = output->Row(input_rows-1);
    ConvoluteStandardSymmetric(input->Row(input_rows-2), output_oR, input_cols, kernel->Row(0));
    ConvoluteStandardSymmetric(input->Row(input_rows-1), output_oR, input_cols, kernel->Row(1));
}

void ConvoluteStandard2D2DSymmetric3(matrix* input, matrix* output, matrix* kernel){
    int input_rows = input->rows, input_cols = input->cols;

    double* output_oR = output->Row(0);
    ConvoluteStandardSymmetric(input->Row(0), output_oR, input_cols, kernel->elem + 1);
    ConvoluteStandardSymmetric(input->Row(1), output_oR, input_cols, kernel->elem);

    for(int oR=1; oR<input_rows-1; ++oR){
        output_oR = output->Row(oR);
        ConvoluteStandardSymmetric(input->Row(oR-1), output_oR, input_cols, kernel->elem);
        ConvoluteStandardSymmetric(input->Row(oR),   output_oR, input_cols, kernel->elem + 1);
        ConvoluteStandardSymmetric(input->Row(oR+1), output_oR, input_cols, kernel->elem);
    }

    output_oR = output->Row(input_rows-1);
    ConvoluteStandardSymmetric(input->Row(input_rows-2), output_oR, input_cols, kernel->elem);
    ConvoluteStandardSymmetric(input->Row(input_rows-1), output_oR, input_cols, kernel->elem + 1);
}


void PyramidalConvolution2D2D(matrix* input, matrix* output, matrix* kernel, int stair){
    int input_rows = input->rows, input_cols = input->cols;
    double* output_oR;
    for(int oR = 1 + stair; oR < input_rows - (1 + stair); ++oR){
        output_oR = output->Row(oR);
        ConvoluteSymmetricNoPadding(input->Row(oR-1) + stair, output_oR + stair, input_cols - 2 * stair, kernel->Row(0));
        ConvoluteSymmetricNoPadding(input->Row(oR)   + stair, output_oR + stair, input_cols - 2 * stair, kernel->Row(1));
        ConvoluteSymmetricNoPadding(input->Row(oR+1) + stair, output_oR + stair, input_cols - 2 * stair, kernel->Row(2));
    }
}



void ConvoluteStandard2D2DFullySymmetric(matrix* input, matrix* output, matrix* kernel){
    int input_rows = input->rows, input_cols = input->cols;

    double* output_oR = output->Row(0);
    ConvoluteStandardSymmetric(input->Row(0), output_oR, input_cols, kernel->elem);
    ConvoluteStandardFullySymmetric0(input->Row(1), output_oR, input_cols, kernel->elem);

    for(int oR=1; oR<input_rows-1; ++oR){
        output_oR = output->Row(oR);
        ConvoluteStandardFullySymmetric0(input->Row(oR-1), output_oR, input_cols, kernel->elem);
        ConvoluteStandardSymmetric(input->Row(oR),   output_oR, input_cols, kernel->elem);
        ConvoluteStandardFullySymmetric0(input->Row(oR+1), output_oR, input_cols, kernel->elem);
    }

    output_oR = output->Row(input_rows-1);
    ConvoluteStandardFullySymmetric0(input->Row(input_rows-2), output_oR, input_cols, kernel->elem);
    ConvoluteStandardSymmetric(input->Row(input_rows-1), output_oR, input_cols, kernel->elem);
}


void BackwardConvoluteStandard(double* input, double* inputDelta, double* outputDelta, double* kernel, double* kernelGrad, const int siz){
    double input_iR_0 = input[0], input_iR_1 = input[1], input_iR_2 = input[2], input_iR_3 = input[3];
    kernelGrad[2] +=                                outputDelta[0] * input_iR_1 + outputDelta[1] * input_iR_2 + outputDelta[2] * input_iR_3;
    kernelGrad[1] += outputDelta[0] * input_iR_0 +  outputDelta[1] * input_iR_1 + outputDelta[2] * input_iR_2 + outputDelta[3] * input_iR_3;
    kernelGrad[0] += outputDelta[1] * input_iR_0 +  outputDelta[2] * input_iR_1 + outputDelta[3] * input_iR_2 + outputDelta[4] * input_iR_3;

    for(int iR=4; iR<siz-4; iR+=4){
        input_iR_0 = input[iR]; input_iR_1 = input[iR+1]; input_iR_2 = input[iR+2]; input_iR_3 = input[iR+3];
        kernelGrad[2] += outputDelta[iR-1] * input_iR_0 + outputDelta[iR  ] * input_iR_1 + outputDelta[iR+1] * input_iR_2 + outputDelta[iR+2] * input_iR_3;
        kernelGrad[1] += outputDelta[iR  ] * input_iR_0 + outputDelta[iR+1] * input_iR_1 + outputDelta[iR+2] * input_iR_2 + outputDelta[iR+3] * input_iR_3;
        kernelGrad[0] += outputDelta[iR+1] * input_iR_0 + outputDelta[iR+2] * input_iR_1 + outputDelta[iR+3] * input_iR_2 + outputDelta[iR+4] * input_iR_3;
    }

    input_iR_0 = input[siz-4]; input_iR_1 = input[siz-3]; input_iR_2 = input[siz-2]; input_iR_3 = input[siz-1];
    kernelGrad[2] += outputDelta[siz-5] * input_iR_0 + outputDelta[siz-4] * input_iR_1 + outputDelta[siz-3] * input_iR_2 + outputDelta[siz-2] * input_iR_3;
    kernelGrad[1] += outputDelta[siz-4] * input_iR_0 + outputDelta[siz-3] * input_iR_1 + outputDelta[siz-2] * input_iR_2 + outputDelta[siz-1] * input_iR_3;
    kernelGrad[0] += outputDelta[siz-3] * input_iR_0 + outputDelta[siz-2] * input_iR_1 + outputDelta[siz-1] * input_iR_2;

    inputDelta[0] += outputDelta[0] * kernel[1] + outputDelta[1] * kernel[0];
    for(int iR=1; iR<siz-1; ++iR)
        inputDelta[iR  ] += outputDelta[iR-1] * kernel[2] + outputDelta[iR  ] * kernel[1] + outputDelta[iR+1] * kernel[0];
    inputDelta[siz-1] += outputDelta[siz-2] * kernel[2] + outputDelta[siz-1] * kernel[1];
}



void BackwardConvoluteStandardSymmetric(double* input, double* inputDelta, double* outputDelta, double* kernel, double* kernelGrad, const int siz){
    double input_iR_0 = input[0], input_iR_1 = input[1], input_iR_2 = input[2], input_iR_3 = input[3];

    kernelGrad[0] += outputDelta[1] * input_iR_0 +  (outputDelta[0] + outputDelta[2]) * input_iR_1 +
    (outputDelta[1] + outputDelta[3]) * input_iR_2 + (outputDelta[2] + outputDelta[4]) * input_iR_3;
    kernelGrad[1] += outputDelta[0] * input_iR_0 +  outputDelta[1] * input_iR_1 + outputDelta[2] * input_iR_2 + outputDelta[3] * input_iR_3;

    for(int iR=4; iR<siz-4; iR+=4){
        input_iR_0 = input[iR]; input_iR_1 = input[iR+1]; input_iR_2 = input[iR+2]; input_iR_3 = input[iR+3];

        kernelGrad[0] += (outputDelta[iR-1] + outputDelta[iR+1]) * input_iR_0 + (outputDelta[iR  ] + outputDelta[iR+2]) * input_iR_1 +
        (outputDelta[iR+1] + outputDelta[iR+3]) * input_iR_2 + (outputDelta[iR+2] + outputDelta[iR+4]) * input_iR_3;

        kernelGrad[1] += outputDelta[iR  ] * input_iR_0 + outputDelta[iR+1] * input_iR_1 + outputDelta[iR+2] * input_iR_2 + outputDelta[iR+3] * input_iR_3;
    }

    input_iR_0 = input[siz-4]; input_iR_1 = input[siz-3]; input_iR_2 = input[siz-2]; input_iR_3 = input[siz-1];
    kernelGrad[0] += (outputDelta[siz-5] + outputDelta[siz-3]) * input_iR_0 + (outputDelta[siz-4] + outputDelta[siz-2])* input_iR_1 +
    (outputDelta[siz-3] + outputDelta[siz-1]) * input_iR_2 + outputDelta[siz-2] * input_iR_3;
    kernelGrad[1] += outputDelta[siz-4] * input_iR_0 + outputDelta[siz-3] * input_iR_1 + outputDelta[siz-2] * input_iR_2 + outputDelta[siz-1] * input_iR_3;

    inputDelta[0] += outputDelta[0] * kernel[1] + outputDelta[1] * kernel[0];
    for(int iR=1; iR<siz-1; ++iR)
        inputDelta[iR  ] += (outputDelta[iR-1] + outputDelta[iR+1]) * kernel[0] + outputDelta[iR  ] * kernel[1];
    inputDelta[siz-1] += outputDelta[siz-2] * kernel[0] + outputDelta[siz-1] * kernel[1];
}


void BackwardConvoluteSymmetricNoPadding(double* input, double* inputDelta, double* outputDelta, double* kernel, double* kernelGrad, const int siz){
    for(int oC=1; oC<siz-1; ++oC){
        inputDelta[oC-1] += outputDelta[oC] * kernel[0];
        inputDelta[oC]   += outputDelta[oC] * kernel[1];
        inputDelta[oC+1] += outputDelta[oC] * kernel[0];
        kernelGrad[0] += outputDelta[oC] * (input[oC-1] + input[oC+1]);
        kernelGrad[1] += outputDelta[oC] * input[oC];
    }
}




void BackwardConvoluteStandardFullySymmetric0(double* input, double* inputDelta, double* outputDelta, double* kernel, double* kernelGrad, const int siz){
    kernelGrad[0] += (outputDelta[0] + outputDelta[1]) * input[0];
    for(int iR=1; iR<siz-1; ++iR)
        kernelGrad[0] += (outputDelta[iR-1] + outputDelta[iR] + outputDelta[iR+1]) * input[iR];
    kernelGrad[0] += (outputDelta[siz-2] + outputDelta[siz-1]) * input[siz-1];

    inputDelta[0] += (outputDelta[0] + outputDelta[1]) * kernel[0];
    for(int iR=1; iR<siz-1; ++iR)
        inputDelta[iR  ] += (outputDelta[iR-1] + outputDelta[iR] + outputDelta[iR+1]) * kernel[0];
    inputDelta[siz-1] += (outputDelta[siz-2] + outputDelta[siz-1]) * kernel[0];
}


void BackwardConvoluteStandard_4(double* input, double* inputDelta, double* outputDelta, double* kernel, double* kernelGrad){
    double input_iR_0 = input[0], input_iR_1 = input[1], input_iR_2 = input[2], input_iR_3 = input[3];
    kernelGrad[2] +=                                outputDelta[0] * input_iR_1 + outputDelta[1] * input_iR_2 + outputDelta[2] * input_iR_3;
    kernelGrad[1] += outputDelta[0] * input_iR_0 +  outputDelta[1] * input_iR_1 + outputDelta[2] * input_iR_2 + outputDelta[3] * input_iR_3;
    kernelGrad[0] += outputDelta[1] * input_iR_0 +  outputDelta[2] * input_iR_1 + outputDelta[3] * input_iR_2;

    inputDelta[0] += outputDelta[0] * kernel[1] + outputDelta[1] * kernel[0];
    for(int iR=1; iR<3; ++iR)
        inputDelta[iR  ] += outputDelta[iR-1] * kernel[2] + outputDelta[iR  ] * kernel[1] + outputDelta[iR+1] * kernel[0];
    inputDelta[3] += outputDelta[2] * kernel[2] + outputDelta[3] * kernel[1];
}


void BackwardConvoluteGradStandard_4(double* input, double* outputDelta, double* kernelGrad){
    double input_iR_0 = input[0], input_iR_1 = input[1], input_iR_2 = input[2], input_iR_3 = input[3];
    kernelGrad[2] +=                                outputDelta[0] * input_iR_1 + outputDelta[1] * input_iR_2 + outputDelta[2] * input_iR_3;
    kernelGrad[1] += outputDelta[0] * input_iR_0 +  outputDelta[1] * input_iR_1 + outputDelta[2] * input_iR_2 + outputDelta[3] * input_iR_3;
    kernelGrad[0] += outputDelta[1] * input_iR_0 +  outputDelta[2] * input_iR_1 + outputDelta[3] * input_iR_2;
}




void BackwardConvoluteGradStandard(double* input, double* outputDelta, double* kernelGrad, const int siz){
    double input_iR_0 = input[0], input_iR_1 = input[1], input_iR_2 = input[2], input_iR_3 = input[3];
    kernelGrad[2] +=                                outputDelta[0] * input_iR_1 + outputDelta[1] * input_iR_2 + outputDelta[2] * input_iR_3;
    kernelGrad[1] += outputDelta[0] * input_iR_0 +  outputDelta[1] * input_iR_1 + outputDelta[2] * input_iR_2 + outputDelta[3] * input_iR_3;
    kernelGrad[0] += outputDelta[1] * input_iR_0 +  outputDelta[2] * input_iR_1 + outputDelta[3] * input_iR_2;

    for(int iR=4; iR<siz-4; iR+=4){
        input_iR_0 = input[iR]; input_iR_1 = input[iR+1]; input_iR_2 = input[iR+2]; input_iR_3 = input[iR+3];
        kernelGrad[2] += outputDelta[iR-1] * input_iR_0 + outputDelta[iR  ] * input_iR_1 + outputDelta[iR+1] * input_iR_2 + outputDelta[iR+2] * input_iR_3;
        kernelGrad[1] += outputDelta[iR  ] * input_iR_0 + outputDelta[iR+1] * input_iR_1 + outputDelta[iR+2] * input_iR_2 + outputDelta[iR+3] * input_iR_3;
        kernelGrad[0] += outputDelta[iR+1] * input_iR_0 + outputDelta[iR+2] * input_iR_1 + outputDelta[iR+3] * input_iR_2 + outputDelta[iR+4] * input_iR_3;
    }

    input_iR_0 = input[siz-4]; input_iR_1 = input[siz-3]; input_iR_2 = input[siz-2]; input_iR_3 = input[siz-1];
    kernelGrad[2] += outputDelta[siz-5] * input_iR_0 + outputDelta[siz-4] * input_iR_1 + outputDelta[siz-3] * input_iR_2 + outputDelta[siz-2] * input_iR_3;
    kernelGrad[1] += outputDelta[siz-4] * input_iR_0 + outputDelta[siz-3] * input_iR_1 + outputDelta[siz-2] * input_iR_2 + outputDelta[siz-1] * input_iR_3;
    kernelGrad[0] += outputDelta[siz-3] * input_iR_0 + outputDelta[siz-2] * input_iR_1 + outputDelta[siz-1] * input_iR_2;
}

void BackwardConvoluteStandard2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
                           realNumber* biasGrad){
    if (input->rows ==4 && input->cols ==4){
        BackwardConvoluteStandard2D2D_4_4(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad);
        return;
    }

    if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();
    double* inputDelta_iR = inputDelta->Row(0);
    double* input_iR = input->Row(0);
    int input_rows = input->rows, input_cols = input->cols;
    BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(0), kernel->Row(1), kernelGrad->Row(1), input_cols);
    BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(1), kernel->Row(0), kernelGrad->Row(0), input_cols);
    for(int iR=1; iR<input_rows-1; ++iR){
        input_iR = input->Row(iR);
        inputDelta_iR = inputDelta->Row(iR);
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel->Row(2), kernelGrad->Row(2), input_cols);
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel->Row(1), kernelGrad->Row(1), input_cols);
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel->Row(0), kernelGrad->Row(0), input_cols);
    }
    input_iR = input->Row(input_rows-1);
    inputDelta_iR = inputDelta->Row(input_rows-1);
    BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(input_rows-2), kernel->Row(2), kernelGrad->Row(2), input_cols);
    BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(input_rows-1), kernel->Row(1), kernelGrad->Row(1), input_cols);
}



void BackwardConvoluteStandard2D2DSymmetric(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
                           realNumber* biasGrad){
    if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();
    double* inputDelta_iR = inputDelta->Row(0);
    double* input_iR = input->Row(0);
    int input_rows = input->rows, input_cols = input->cols;
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(0), kernel->Row(1), kernelGrad->Row(1), input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(1), kernel->Row(0), kernelGrad->Row(0), input_cols);
    for(int iR=1; iR<input_rows-1; ++iR){
        input_iR = input->Row(iR);
        inputDelta_iR = inputDelta->Row(iR);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel->Row(2), kernelGrad->Row(2), input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel->Row(1), kernelGrad->Row(1), input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel->Row(0), kernelGrad->Row(0), input_cols);
    }
    input_iR = input->Row(input_rows-1);
    inputDelta_iR = inputDelta->Row(input_rows-1);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-2), kernel->Row(2), kernelGrad->Row(2), input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-1), kernel->Row(1), kernelGrad->Row(1), input_cols);
}



void BackwardConvoluteStandard2D2DSymmetric2(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad){
    if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();
    double* inputDelta_iR = inputDelta->Row(0);
    double* input_iR = input->Row(0);
    int input_rows = input->rows, input_cols = input->cols;
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(0), kernel->Row(1), kernelGrad->Row(1), input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(1), kernel->Row(0), kernelGrad->Row(0), input_cols);
    for(int iR=1; iR<input_rows-1; ++iR){
        input_iR = input->Row(iR);
        inputDelta_iR = inputDelta->Row(iR);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel->Row(0), kernelGrad->Row(0), input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel->Row(1), kernelGrad->Row(1), input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel->Row(0), kernelGrad->Row(0), input_cols);
    }
    input_iR = input->Row(input_rows-1);
    inputDelta_iR = inputDelta->Row(input_rows-1);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-2), kernel->Row(0), kernelGrad->Row(0), input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-1), kernel->Row(1), kernelGrad->Row(1), input_cols);
}



void BackwardConvoluteStandard2D2DSymmetric3(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad){
    if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();
    double* inputDelta_iR = inputDelta->Row(0);
    double* input_iR = input->Row(0);
    int input_rows = input->rows, input_cols = input->cols;
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(0), kernel->elem+1, kernelGrad->elem+1, input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(1), kernel->elem, kernelGrad->elem, input_cols);
    for(int iR=1; iR<input_rows-1; ++iR){
        input_iR = input->Row(iR);
        inputDelta_iR = inputDelta->Row(iR);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel->elem, kernelGrad->elem, input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel->elem+1, kernelGrad->elem+1, input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel->elem, kernelGrad->elem, input_cols);
    }
    input_iR = input->Row(input_rows-1);
    inputDelta_iR = inputDelta->Row(input_rows-1);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-2), kernel->elem, kernelGrad->elem, input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-1), kernel->elem+1, kernelGrad->elem+1, input_cols);
}






void BackwardPyramidalConvolution2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, int stair){
    double* outputDelta_oR;
    int input_rows = input->rows, input_cols = input->cols;
    for(int oR = 1 + stair; oR < input_rows - (1 + stair); ++oR){
        outputDelta_oR = outputDelta->Row(oR);
        BackwardConvoluteSymmetricNoPadding(input->Row(oR-1)+stair, inputDelta->Row(oR-1)+stair, outputDelta_oR+stair, kernel->Row(0), kernelGrad->Row(0), input_cols - 2 * stair);
        BackwardConvoluteSymmetricNoPadding(input->Row(oR)  +stair, inputDelta->Row(oR)  +stair, outputDelta_oR+stair, kernel->Row(1), kernelGrad->Row(1), input_cols - 2 * stair);
        BackwardConvoluteSymmetricNoPadding(input->Row(oR+1)+stair, inputDelta->Row(oR+1)+stair, outputDelta_oR+stair, kernel->Row(2), kernelGrad->Row(2), input_cols - 2 * stair);
    }
}




void BackwardConvoluteStandard2D2DFullySymmetric(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
                           realNumber* biasGrad){
    if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();
    double* inputDelta_iR = inputDelta->Row(0);
    double* input_iR = input->Row(0);
    int input_rows = input->rows, input_cols = input->cols;
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(0), kernel->elem, kernelGrad->elem, input_cols);
    BackwardConvoluteStandardFullySymmetric0(input_iR, inputDelta_iR, outputDelta->Row(1), kernel->elem, kernelGrad->elem, input_cols);
    for(int iR=1; iR<input_rows-1; ++iR){
        input_iR = input->Row(iR);
        inputDelta_iR = inputDelta->Row(iR);
        BackwardConvoluteStandardFullySymmetric0(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel->elem, kernelGrad->elem, input_cols);
        BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel->elem, kernelGrad->elem, input_cols);
        BackwardConvoluteStandardFullySymmetric0(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel->elem, kernelGrad->elem, input_cols);
    }
    input_iR = input->Row(input_rows-1);
    inputDelta_iR = inputDelta->Row(input_rows-1);
    BackwardConvoluteStandardFullySymmetric0(input_iR, inputDelta_iR, outputDelta->Row(input_rows-2), kernel->elem, kernelGrad->elem, input_cols);
    BackwardConvoluteStandardSymmetric(input_iR, inputDelta_iR, outputDelta->Row(input_rows-1), kernel->elem, kernelGrad->elem, input_cols);
}



void BackwardConvoluteGradStandard2D2D(matrix* input, matrix* outputDelta, matrix* kernelGrad, realNumber* biasGrad){
    if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();
    double* input_iR = input->Row(0);
    int input_rows = input->rows, input_cols = input->cols;
    BackwardConvoluteGradStandard(input_iR, outputDelta->Row(0), kernelGrad->Row(1), input_cols);
    BackwardConvoluteGradStandard(input_iR, outputDelta->Row(1), kernelGrad->Row(0), input_cols);
    for(int iR=1; iR<input_rows-1; ++iR){
        input_iR = input->Row(iR);
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(iR-1), kernelGrad->Row(2), input_cols);
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(iR),   kernelGrad->Row(1), input_cols);
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(iR+1), kernelGrad->Row(0), input_cols);
    }
    input_iR = input->Row(input_rows-1);
    BackwardConvoluteGradStandard(input_iR, outputDelta->Row(input_rows-2), kernelGrad->Row(2), input_cols);
    BackwardConvoluteGradStandard(input_iR, outputDelta->Row(input_rows-1), kernelGrad->Row(1), input_cols);
}



void BackwardConvolute2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

        int minInputRow, maxInputRow, minInputCol, maxInputCol, CC, indexOutputSize;
        double* inputDelta_iR, *outputDelta_oR, *input_iR;
        double outputDelta_oR_oC;
        const double eps = 1E-10;
        double *kernel_RR, *kernelGrad_RR;
        int kR = kernel->rows, kC = kernel->cols;
        int minR = kR - 1 - paddingR, minC =  kC - 1 - paddingC;
        int inp_rows_1 = input->rows - 1, inp_cols_1 = input->cols - 1;
        int temp_oR;

        if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();

        for(int oR=0; oR<outputDelta->rows; ++oR){
            temp_oR = oR - paddingR;
            outputDelta_oR = outputDelta->Row(oR);

            indexOutputRow.resize(0);
            for(int oC=0; oC<outputDelta->cols; ++oC)
                if (fabs(outputDelta_oR[oC])>eps)
                    indexOutputRow.push_back(oC);
            indexOutputSize = indexOutputRow.size();

            minInputRow = max(0, temp_oR);
            maxInputRow = min(inp_rows_1, oR + minR);

            for(int iR = minInputRow; iR<=maxInputRow; ++iR){
                inputDelta_iR = inputDelta->Row(iR);
                input_iR = input->Row(iR);
                kernel_RR     = kernel    ->Row(iR-temp_oR);
                kernelGrad_RR = kernelGrad->Row(iR-temp_oR);

                for(int oCIndex=0, oC; oCIndex<indexOutputSize; ++oCIndex){
                    oC = indexOutputRow[oCIndex];
                    outputDelta_oR_oC = outputDelta_oR[oC];
                    minInputCol = max(0, oC - paddingC);
                    maxInputCol = min(inp_cols_1, oC + minC);
                    CC = paddingC - oC;
                    for(int iC=minInputCol; iC<=maxInputCol; ++iC){
                        inputDelta_iR[iC] += kernel_RR[CC+iC] * outputDelta_oR_oC;
                        kernelGrad_RR[iC+CC] += outputDelta_oR_oC * input_iR[iC];
                        //outputDelta->At(oR,oC) += input->At(iR, iC)*kernel->At(iR-oR+paddingR, iC-oC+paddingC);
                    }
                }
            }
        }
}



void BackwardConvoluteGrad2D2D(matrix* input, matrix* outputDelta, matrix* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow){

        int minInputRow, maxInputRow, minInputCol, maxInputCol, CC, indexOutputSize;
        double *outputDelta_oR, *input_iR, *kernelGrad_RR;
        double outputDelta_oR_oC;
        const double eps = 1E-10;
        int kR = kernelGrad->rows, kC = kernelGrad->cols;
        int minR = kR - 1 - paddingR, minC =  kC - 1 - paddingC;
        int inp_rows_1 = input->rows - 1, inp_cols_1 = input->cols - 1;
        int temp_oR;

        if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();

        for(int oR=0; oR<outputDelta->rows; ++oR){
            temp_oR = oR - paddingR;
            outputDelta_oR = outputDelta->Row(oR);

            indexOutputRow.resize(0);
            for(int oC=0; oC<outputDelta->cols; ++oC)
                if (fabs(outputDelta_oR[oC])>eps)
                    indexOutputRow.push_back(oC);
            indexOutputSize = indexOutputRow.size();

            minInputRow = max(0, temp_oR);
            maxInputRow = min(inp_rows_1, oR + minR);

            for(int iR = minInputRow; iR<=maxInputRow; ++iR){
                input_iR = input->Row(iR);
                kernelGrad_RR = kernelGrad->Row(iR-temp_oR);

                for(int oCIndex=0, oC; oCIndex<indexOutputSize; ++oCIndex){
                    oC = indexOutputRow[oCIndex];
                    outputDelta_oR_oC = outputDelta_oR[oC];
                    minInputCol = max(0, oC - paddingC);
                    maxInputCol = min(inp_cols_1, oC + minC);
                    CC = paddingC - oC;
                    for(int iC=minInputCol; iC<=maxInputCol; ++iC){
                        kernelGrad_RR[CC+iC] += outputDelta_oR_oC * input_iR[iC];
                    }
                }
            }
        }
}




void ConvoluteQuadratic2D2D(matrix* input, matrix* output, matrix* linearReversedKernel, matrix* quadraticReversedKernel,
                            realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow){

    double * input_iR, *output_oR, *linRevKernel_iR__oR, *quadRevKernel_iR__oR, *output_elem;
    double   input_iR_iC;
    const double eps = 1E-10;
    int tempInd, indexInputSize, minOutRow, maxOutRow, minOutCol, maxOutCol;
    int kR = linearReversedKernel->rows, kC = linearReversedKernel->cols;
    int input_rows = input->rows, input_cols = input->cols;
    int maxR = - kR + 1 + paddingR, maxC = - kC + 1 + paddingC;
    int minR = input_rows - kR + 2 * paddingR, minC = input_cols - kC + 2*paddingC;
    int tempC = kC - paddingC - 1;
    int temp_iR;
    output_elem = output->elem;

    if (bias){
        double bias_ = bias->elem[0];
        for(int j=0; j<output->len; ++j)
            output_elem[j] += bias_;
    }

    for(int iR=0; iR<input_rows; ++iR){
        input_iR = input->Row(iR);
        //build the map of input activity units
        indexInputRow.resize(0);
        for(int iC=0; iC<input_cols; ++iC)
            if (fabs(input_iR[iC])>eps)
                indexInputRow.push_back(iC);

        indexInputSize = indexInputRow.size();
        temp_iR = maxR + iR;
        minOutRow = max(0, temp_iR);
        maxOutRow = min(iR+paddingR, minR);


        for(int oR=minOutRow; oR<=maxOutRow; ++oR){
            output_oR = output->Row(oR);
            //revKernel_iR__oR = reversedKernel->Row(kR - (iR + paddingR -oR) - 1);
            linRevKernel_iR__oR  = linearReversedKernel     ->Row(oR - temp_iR);//kR - (iR + paddingR -oR) - 1);
            quadRevKernel_iR__oR = quadraticReversedKernel  ->Row(oR - temp_iR);

            for(int iCIndex=0, iC; iCIndex<indexInputSize; ++iCIndex){
                iC = indexInputRow[iCIndex];
                minOutCol = max(0, iC + maxC);
                maxOutCol = min(iC + paddingC, minC);
                input_iR_iC = input_iR[iC];
                tempInd = tempC - iC;
                for(int oC=minOutCol; oC<=maxOutCol; ++oC){
                    output_oR[oC]+=input_iR_iC * (linRevKernel_iR__oR[tempInd+oC] + quadRevKernel_iR__oR[tempInd+oC] * input_iR_iC);//    kernel_iR__oR[iC-oC];
                }
            }
        }
    }
}


void BackwardConvoluteQuadratic2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* linearKernel, matrix* quadraticKernel,
                                    matrix* linearKernelGrad, matrix* quadraticKernelGrad, realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

        int minInputRow, maxInputRow, minInputCol, maxInputCol, CC, indexOutputSize;
        double* inputDelta_iR, *outputDelta_oR, *input_iR, *linearKernel_RR, *quadraticKernel_RR, *linearKernelGrad_RR, *quadraticKernelGrad_RR;
        double outputDelta_oR_oC;
        const double eps = 1E-10;
        int kR = linearKernel->rows, kC = linearKernel->cols;
        int minR = kR - 1 - paddingR, minC =  kC - 1 - paddingC;
        int inp_rows_1 = input->rows - 1, inp_cols_1 = input->cols - 1;
        int temp_oR;

        if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();

        for(int oR=0; oR<outputDelta->rows; ++oR){
            temp_oR = oR - paddingR;
            outputDelta_oR = outputDelta->Row(oR);

            indexOutputRow.resize(0);
            for(int oC=0; oC<outputDelta->cols; ++oC)
                if (fabs(outputDelta_oR[oC])>eps)
                    indexOutputRow.push_back(oC);
            indexOutputSize = indexOutputRow.size();

            minInputRow = max(0, temp_oR);
            maxInputRow = min(inp_rows_1, oR + minR);

            for(int iR = minInputRow; iR<=maxInputRow; ++iR){
                inputDelta_iR = inputDelta->Row(iR);
                input_iR = input->Row(iR);

                linearKernel_RR    = linearKernel   ->Row(iR-temp_oR);
                quadraticKernel_RR = quadraticKernel->Row(iR-temp_oR);

                linearKernelGrad_RR     = linearKernelGrad      ->Row(iR-temp_oR);
                quadraticKernelGrad_RR  = quadraticKernelGrad   ->Row(iR-temp_oR);

                for(int oCIndex=0, oC; oCIndex<indexOutputSize; ++oCIndex){
                    oC = indexOutputRow[oCIndex];
                    outputDelta_oR_oC = outputDelta_oR[oC];
                    minInputCol = max(0, oC - paddingC);
                    maxInputCol = min(inp_cols_1, oC + minC);
                    CC = paddingC - oC;

                    for(int iC=minInputCol; iC<=maxInputCol; ++iC){
                        inputDelta_iR[iC] += outputDelta_oR_oC * (linearKernel_RR[CC+iC] + 2.0 * quadraticKernel_RR[CC+iC] * input_iR[iC]);
                        linearKernelGrad_RR[CC+iC] += outputDelta_oR_oC * input_iR[iC];
                        quadraticKernelGrad_RR[CC+iC] += outputDelta_oR_oC * sqr(input_iR[iC]);

//                        inputDelta_iR[iC] += kernel_RR[CC+iC] * outputDelta_oR_oC;
//                        kernelGrad_RR[CC+iC] += outputDelta_oR_oC * input_iR[iC];
                        //outputDelta->At(oR,oC) += input->At(iR, iC)*kernel->At(iR-oR+paddingR, iC-oC+paddingC);
                    }
                }
            }
        }
}


void BackwardConvoluteQuadraticGrad2D2D(matrix* input, matrix* outputDelta, matrix* linearKernelGrad, matrix* quadraticKernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow){

        int minInputRow, maxInputRow, minInputCol, maxInputCol, CC, indexOutputSize;
        double *outputDelta_oR, *input_iR, *linearKernelGrad_RR, *quadraticKernelGrad_RR;
        double outputDelta_oR_oC;
        const double eps = 1E-10;
        int kR = linearKernelGrad->rows, kC = linearKernelGrad->cols;
        int minR = kR - 1 - paddingR, minC =  kC - 1 - paddingC;
        int inp_rows_1 = input->rows - 1, inp_cols_1 = input->cols - 1;
        int temp_oR;

        if (biasGrad)
            biasGrad->elem[0]+=outputDelta->Sum();

        for(int oR=0; oR<outputDelta->rows; ++oR){
            temp_oR = oR - paddingR;
            outputDelta_oR = outputDelta->Row(oR);

            indexOutputRow.resize(0);
            for(int oC=0; oC<outputDelta->cols; ++oC)
                if (fabs(outputDelta_oR[oC])>eps)
                    indexOutputRow.push_back(oC);
            indexOutputSize = indexOutputRow.size();

            minInputRow = max(0, temp_oR);
            maxInputRow = min(inp_rows_1, oR + minR);

            for(int iR = minInputRow; iR<=maxInputRow; ++iR){
                input_iR = input->Row(iR);

                linearKernelGrad_RR     = linearKernelGrad      ->Row(iR-temp_oR);
                quadraticKernelGrad_RR  = quadraticKernelGrad   ->Row(iR-temp_oR);

                for(int oCIndex=0, oC; oCIndex<indexOutputSize; ++oCIndex){
                    oC = indexOutputRow[oCIndex];
                    outputDelta_oR_oC = outputDelta_oR[oC];
                    minInputCol = max(0, oC - paddingC);
                    maxInputCol = min(inp_cols_1, oC + minC);
                    CC = paddingC - oC;

                    for(int iC=minInputCol; iC<=maxInputCol; ++iC){
                           linearKernelGrad_RR[CC+iC] += outputDelta_oR_oC *     input_iR[iC];
                        quadraticKernelGrad_RR[CC+iC] += outputDelta_oR_oC * sqr(input_iR[iC]);
                        //kernelGrad_RR[CC+iC] += outputDelta_oR_oC * input_iR[iC];
                    }
                }
            }
        }
}



void Convolute2D3D(matrix* input, tensor* output, tensor* reversedKernel, vect* bias, int paddingR, int paddingC, vector<int> & indexInputRow){
    matrix* output_d = new matrix();
    matrix* reversedKernel_d = new matrix();
    realNumber* bias_d = new realNumber();

    for(int d=0; d<reversedKernel->depth; d++){
        output_d->SetToTensorLayer(output, d);
        reversedKernel_d->SetToTensorLayer(reversedKernel, d);
        bias_d  ->SetToVectElement(bias, d);

        Convolute2D2D(input, output_d, reversedKernel_d, bias_d, paddingR, paddingC, indexInputRow);
    }
    //deleting only shell without deallocation memory
    DeleteOnlyShell(output_d);
    DeleteOnlyShell(reversedKernel_d);
    DeleteOnlyShell(bias_d);
}



void BackwardConvolute2D3D(matrix* input, matrix* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

    matrix* outputDelta_d = new matrix();
    matrix* kernel_d = new matrix();
    matrix* kernelGrad_d = new matrix();
    realNumber* biasGrad_d = new realNumber();

    for(int d=0; d<kernel->depth; d++){
        outputDelta_d->SetToTensorLayer(outputDelta, d);
        kernel_d->SetToTensorLayer(kernel, d);
        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
        biasGrad_d  ->SetToVectElement(biasGrad, d);

        BackwardConvolute2D2D(input, inputDelta, outputDelta_d, kernel_d, kernelGrad_d, biasGrad_d, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(outputDelta_d);
    DeleteOnlyShell(kernel_d);
    DeleteOnlyShell(kernelGrad_d);
    DeleteOnlyShell(biasGrad_d);
}


void BackwardConvoluteGrad2D3D(matrix* input, tensor* outputDelta, tensor* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

    matrix* outputDelta_d = new matrix();
    matrix* kernelGrad_d = new matrix();
    realNumber* biasGrad_d = new realNumber();

    for(int d=0; d<kernelGrad->depth; d++){
        outputDelta_d->SetToTensorLayer(outputDelta, d);
        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
        biasGrad_d  ->SetToVectElement(biasGrad, d);

        BackwardConvoluteGrad2D2D(input, outputDelta_d, kernelGrad_d, biasGrad_d, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(outputDelta_d);
    DeleteOnlyShell(kernelGrad_d);
    DeleteOnlyShell(biasGrad_d);
}


void ConvoluteStandard3D2D_2_2(tensor* input, matrix* output, tensor* kernel, realNumber* bias){
    double* output_ = output->elem;
    double* input_d, *kernel_d;

    output->Add(bias->elem[0]);
    for(int d=0; d<kernel->depth; ++d){
        input_d = input->Layer(d);
        kernel_d = kernel->Layer(d);
        output_[0] += input_d[0] * kernel_d[4] + input_d[1] * kernel_d[5] + input_d[2] * kernel_d[7] + input_d[3] * kernel_d[8];
        output_[1] += input_d[0] * kernel_d[3] + input_d[1] * kernel_d[4] + input_d[2] * kernel_d[6] + input_d[3] * kernel_d[7];
        output_[2] += input_d[0] * kernel_d[1] + input_d[1] * kernel_d[2] + input_d[2] * kernel_d[4] + input_d[3] * kernel_d[5];
        output_[3] += input_d[0] * kernel_d[0] + input_d[1] * kernel_d[1] + input_d[2] * kernel_d[3] + input_d[3] * kernel_d[4];
    }
}


void BackwardConvoluteStandard3D2D_2_2(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, realNumber* biasGrad){
    biasGrad->elem[0]+=outputDelta->Sum();
    double * outputDelta_ = outputDelta->elem;
    double * inputDelta_d;
    double* kernel_d;
    double* kernelGrad_d;
    double* input_d;
    for(int d=0; d<kernel->depth; ++d){
        inputDelta_d = inputDelta->Layer(d);
        kernel_d = kernel->Layer(d);
        inputDelta_d[0] += outputDelta_[0] * kernel_d[4] + outputDelta_[1] * kernel_d[3] + outputDelta_[2] * kernel_d[1] + outputDelta_[3] * kernel_d[0];
        inputDelta_d[1] += outputDelta_[0] * kernel_d[5] + outputDelta_[1] * kernel_d[4] + outputDelta_[2] * kernel_d[2] + outputDelta_[3] * kernel_d[1];
        inputDelta_d[2] += outputDelta_[0] * kernel_d[7] + outputDelta_[1] * kernel_d[6] + outputDelta_[2] * kernel_d[4] + outputDelta_[3] * kernel_d[3];
        inputDelta_d[3] += outputDelta_[0] * kernel_d[8] + outputDelta_[1] * kernel_d[7] + outputDelta_[2] * kernel_d[5] + outputDelta_[3] * kernel_d[4];
    }

    for(int d=0; d<kernel->depth; ++d){
        kernelGrad_d = kernelGrad->Layer(d);
        input_d = input->Layer(d);
        kernelGrad_d[0] += outputDelta_[3] * input_d[0];
        kernelGrad_d[1] += outputDelta_[2] * input_d[0] + outputDelta_[3] * input_d[1];
        kernelGrad_d[2] += outputDelta_[2] * input_d[1];
        kernelGrad_d[3] += outputDelta_[1] * input_d[0] + outputDelta_[3] * input_d[2];
        kernelGrad_d[4] += outputDelta_[0] * input_d[0] + outputDelta_[1] * input_d[1] + outputDelta_[2] * input_d[2] + outputDelta_[3] * input_d[3];
        kernelGrad_d[5] += outputDelta_[0] * input_d[1] + outputDelta_[2] * input_d[3];
        kernelGrad_d[6] += outputDelta_[1] * input_d[2];
        kernelGrad_d[7] += outputDelta_[0] * input_d[2] + outputDelta_[1] * input_d[3];
        kernelGrad_d[8] += outputDelta_[0] * input_d[3];
    }
}


void BackwardConvoluteGradStandard3D2D_2_2(tensor* input, matrix* outputDelta, tensor* kernelGrad, realNumber* biasGrad){
    biasGrad->elem[0]+=outputDelta->Sum();
    double* input_d;
    double * outputDelta_ = outputDelta->elem;
    double* kernelGrad_d;
    for(int d=0; d<kernelGrad->depth; ++d){
        kernelGrad_d = kernelGrad->Layer(d);
        input_d = input->Layer(d);
        kernelGrad_d[0] += outputDelta_[3] * input_d[0];
        kernelGrad_d[1] += outputDelta_[2] * input_d[0] + outputDelta_[3] * input_d[1];
        kernelGrad_d[2] += outputDelta_[2] * input_d[1];
        kernelGrad_d[3] += outputDelta_[1] * input_d[0] + outputDelta_[3] * input_d[2];
        kernelGrad_d[4] += outputDelta_[0] * input_d[0] + outputDelta_[1] * input_d[1] + outputDelta_[2] * input_d[2] + outputDelta_[3] * input_d[3];
        kernelGrad_d[5] += outputDelta_[0] * input_d[1] + outputDelta_[2] * input_d[3];
        kernelGrad_d[6] += outputDelta_[1] * input_d[2];
        kernelGrad_d[7] += outputDelta_[0] * input_d[2] + outputDelta_[1] * input_d[3];
        kernelGrad_d[8] += outputDelta_[0] * input_d[3];
    }
}

void ConvolutionStandardVertical1D(tensor* input, tensor* kernel, matrix* output){
    matrix* input_d = new matrix();

    input_d->SetToTensorLayer(input, 0);
    output->CopyMultiplied(kernel->elem[0], input_d);

    for(int d=1; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output->Add(kernel->elem[d], input_d);
    }
    DeleteOnlyShell(input_d);
}


void ConvolutionStandardVerticalRandom1D(tensor* input, int* indices, tensor* kernel, matrix* output){
    matrix* input_d = new matrix();

    input_d->SetToTensorLayer(input, indices[0]);
    output->CopyMultiplied(kernel->elem[0], input_d);

    for(int d=1; d<kernel->depth; ++d){
        input_d->SetToTensorLayer(input, indices[d]);
        output->Add(kernel->elem[d], input_d);
    }
    DeleteOnlyShell(input_d);
}


void PyramidalVerticalConvolution(tensor* input, int* indices, tensor* kernel, matrix* output, int stair){
    matrix* input_d = new matrix();

    input_d->SetToTensorLayer(input, indices[0]);
    output->CopySubMatrixMultiplied(kernel->elem[0], input_d, stair);

    for(int d=1; d<kernel->depth; ++d){
        input_d->SetToTensorLayer(input, indices[d]);
        output->AddSubMatrix(kernel->elem[d], input_d, stair);
    }
    DeleteOnlyShell(input_d);
}


void AddConvolutionStandardVertical1D(tensor* input, tensor* kernel, matrix* output){
    matrix* input_d = new matrix();

    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output->Add(kernel->elem[d], input_d);
    }

    DeleteOnlyShell(input_d);
}


void BackwardVertical2D(matrix* input, matrix* inputDelta, matrix* outputDelta, double kernel, double & kernelGrad){
    double* outputDelta_elem = outputDelta->elem;
    double* inputDelta_elem = inputDelta->elem;
    double* input_elem = input->elem;
    int input_len = input->len;
    for(int j=0; j<input_len; ++j){
        inputDelta_elem[j] += kernel * outputDelta_elem[j];
    }

    for(int j=0; j<input_len; ++j){
        kernelGrad += input_elem[j] * outputDelta_elem[j];
    }
}


void BackwardConvolutionStandardVertical1D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad){
    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        inputDelta_d->SetToTensorLayer(inputDelta, d);

        //BackwardVertical2D(input_d, inputDelta_d, outputDelta, kernel->elem[d], kernelGrad->elem[d]);

        inputDelta_d->Add(kernel->elem[d], outputDelta);
        kernelGrad->elem[d] += InnerProduct(outputDelta, input_d);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);
}


void BackwardConvolutionStandardVerticalRandom1D(tensor* input, int * indices, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad){
    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();
//    cout<<"Backw conv indices: ";
//    for(int j=0; j<kernel->depth; ++j)
//        cout<<indices[j]<<" ";
//    cout<<endl;
    for(int d=0; d<kernel->depth; ++d){
        input_d->SetToTensorLayer(input, indices[d]);
        inputDelta_d->SetToTensorLayer(inputDelta, indices[d]);

        //BackwardVertical2D(input_d, inputDelta_d, outputDelta, kernel->elem[d], kernelGrad->elem[d]);
        if (indices[d] >= INPUT_DEPTH)
            inputDelta_d->Add(kernel->elem[d], outputDelta);
        kernelGrad->elem[d] += InnerProduct(outputDelta, input_d);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);
}




void BackwardPyramidalVerticalConvolution(tensor* input, int* indices, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, int stair){
    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();
    for(int d=0; d<kernel->depth; ++d){
        input_d->SetToTensorLayer(input, indices[d]);
        inputDelta_d->SetToTensorLayer(inputDelta, indices[d]);

        if (indices[d] >= INPUT_DEPTH)
            inputDelta_d->AddSubMatrix(kernel->elem[d], outputDelta, stair);
        kernelGrad->elem[d] += InnerProductSubMatrices(outputDelta, input_d, stair);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);
}




void BackwardConvolutionStandardVertical1DPartialGrad(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, int startDepth){
    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();

    for(int d=0; d<startDepth; ++d){
        input_d->SetToTensorLayer(input, d);
        kernelGrad->elem[d] += InnerProduct(outputDelta, input_d);
    }

    for(int d=startDepth; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        inputDelta_d->SetToTensorLayer(inputDelta, d);

        inputDelta_d->Add(kernel->elem[d], outputDelta);
        kernelGrad->elem[d] += InnerProduct(outputDelta, input_d);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);
}


void BackwardConvolutionStandardVertical1DGrad(tensor* input, matrix* outputDelta, tensor* kernelGrad){
    matrix* input_d = new matrix();
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        kernelGrad->elem[d] += InnerProduct(outputDelta, input_d);
    }
    DeleteOnlyShell(input_d);
}



void BottleneckConvolutionStandard(tensor* input, matrix* vertOutput, matrix* convOutput, tensor* kernel_vert, matrix* kernel_hor, realNumber* bias_hor){
    ConvolutionStandardVertical1D(input, kernel_vert, vertOutput);
    convOutput->Add(bias_hor->elem[0]);
    ConvoluteStandard2D2D(vertOutput, convOutput, kernel_hor);
}


void BottleneckConvolutionStandardRandom(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, realNumber* biasHor){
    ConvolutionStandardVerticalRandom1D(input, indices, kernelVert, vertOutput);
    convOutput->Add(biasHor->elem[0]);
    ConvoluteStandard2D2D(vertOutput, convOutput, kernelHor);
}

void BottleneckConvolutionStandardRandomSymmetric(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, realNumber* biasHor){
    ConvolutionStandardVerticalRandom1D(input, indices, kernelVert, vertOutput);
    convOutput->Add(biasHor->elem[0]);
    ConvoluteStandard2D2DSymmetric(vertOutput, convOutput, kernelHor);
}

void BottleneckConvolutionStandardRandomSymmetricDrop(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert,
                                                  matrix* kernelHor, realNumber* biasHor, activityData* vertOutputActivity, bool testMode){
    ConvolutionStandardVerticalRandom1D(input, indices, kernelVert, vertOutput);
    if (!testMode)
        vertOutput->SetDroppedElementsToZero(vertOutputActivity);
    convOutput->Add(biasHor->elem[0]);
    ConvoluteStandard2D2DSymmetric(vertOutput, convOutput, kernelHor);
}


void BottleneckConvolutionStandardRandomSymmetricNoBiasDrop(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert,
                                                  matrix* kernelHor, activityData* vertOutputActivity, bool testMode){
    ConvolutionStandardVerticalRandom1D(input, indices, kernelVert, vertOutput);
    if (!testMode)
        vertOutput->SetDroppedElementsToZero(vertOutputActivity);
    ConvoluteStandard2D2DSymmetric(vertOutput, convOutput, kernelHor);
}



void SymmetricConvolution(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, int symmetricLevel){
    ConvolutionStandardVerticalRandom1D(input, indices, kernelVert, vertOutput);
    if (symmetricLevel == 0)
        ConvoluteStandard2D2D(vertOutput, convOutput, kernelHor);
    if (symmetricLevel == 1)
        ConvoluteStandard2D2DSymmetric(vertOutput, convOutput, kernelHor);
    if (symmetricLevel == 2)
        ConvoluteStandard2D2DSymmetric2(vertOutput, convOutput, kernelHor);
    if (symmetricLevel == 3)
        ConvoluteStandard2D2DSymmetric3(vertOutput, convOutput, kernelHor);
    if (symmetricLevel == 4)
        ConvoluteStandard2D2DFullySymmetric(vertOutput, convOutput, kernelHor);
}


void SymmetricConvolution3D(tensor* input, tensor* output, tensor* kernel, vect* bias, int symmetryLevel){
    matrix * input_inp = new matrix();
    matrix * output_out = new matrix();
    matrix * kernel_out_inp = new matrix();
    int kernel_ind = 0;

    for(int out=0; out<output->depth; ++out){
        output_out->SetToTensorLayer(output, out);
        if (bias->len>0)
            output_out->Add(bias->elem[out]);
        for(int inp = 0; inp<input->depth; ++inp){
            input_inp->SetToTensorLayer(input, inp);
            kernel_out_inp->SetToTensorLayer(kernel, kernel_ind);
            ++kernel_ind;

            if (symmetryLevel == 0)
                ConvoluteStandard2D2D(input_inp, output_out, kernel_out_inp);
            if (symmetryLevel == 1)
                ConvoluteStandard2D2DSymmetric(input_inp, output_out, kernel_out_inp);
            if (symmetryLevel == 2)
                ConvoluteStandard2D2DSymmetric2(input_inp, output_out, kernel_out_inp);
            if (symmetryLevel == 3)
                ConvoluteStandard2D2DSymmetric3(input_inp, output_out, kernel_out_inp);
            if (symmetryLevel == 4)
                ConvoluteStandard2D2DFullySymmetric(input_inp, output_out, kernel_out_inp);
        }
    }

    DeleteOnlyShell(input_inp);
    DeleteOnlyShell(output_out);
    DeleteOnlyShell(kernel_out_inp);
}



void BackwardSymmetricConvolution3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad, int symmetryLevel){
    matrix * input_inp = new matrix();
    matrix * inputDelta_inp = new matrix();
    matrix * outputDelta_out = new matrix();
    matrix * kernel_out_inp = new matrix();
    matrix * kernelGrad_out_inp = new matrix();
    int kernel_ind = 0;

    for(int out=0; out<outputDelta->depth; ++out){
        outputDelta_out->SetToTensorLayer(outputDelta, out);
        if (biasGrad->len > 0)
            biasGrad->elem[out] += outputDelta_out->Sum();
        for(int inp = 0; inp<input->depth; ++inp){
            input_inp->SetToTensorLayer(input, inp);
            inputDelta_inp->SetToTensorLayer(inputDelta, inp);
            kernel_out_inp->SetToTensorLayer(kernel, kernel_ind);
            kernelGrad_out_inp->SetToTensorLayer(kernelGrad, kernel_ind);
            ++kernel_ind;

            if (symmetryLevel == 0)
                BackwardConvoluteStandard2D2D(input_inp, inputDelta_inp, outputDelta_out, kernel_out_inp, kernelGrad_out_inp, NULL);
            if (symmetryLevel == 1)
                BackwardConvoluteStandard2D2DSymmetric(input_inp, inputDelta_inp, outputDelta_out, kernel_out_inp, kernelGrad_out_inp, NULL);
            if (symmetryLevel == 2)
                BackwardConvoluteStandard2D2DSymmetric2(input_inp, inputDelta_inp, outputDelta_out, kernel_out_inp, kernelGrad_out_inp, NULL);
            if (symmetryLevel == 3)
                BackwardConvoluteStandard2D2DSymmetric3(input_inp, inputDelta_inp, outputDelta_out, kernel_out_inp, kernelGrad_out_inp, NULL);
            if (symmetryLevel == 4)
                BackwardConvoluteStandard2D2DFullySymmetric(input_inp, inputDelta_inp, outputDelta_out, kernel_out_inp, kernelGrad_out_inp, NULL);
        }
    }

    DeleteOnlyShell(input_inp);
    DeleteOnlyShell(inputDelta_inp);
    DeleteOnlyShell(outputDelta_out);
    DeleteOnlyShell(kernel_out_inp);
    DeleteOnlyShell(kernelGrad_out_inp);
}





void PyramidalConvolution(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, int stair){
    PyramidalVerticalConvolution(input, indices, kernelVert, vertOutput, stair);
    PyramidalConvolution2D2D(vertOutput, convOutput, kernelHor, stair);
}



void BottleneckConvolutionStandardRandomFullySymmetric(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, realNumber* biasHor){
    ConvolutionStandardVerticalRandom1D(input, indices, kernelVert, vertOutput);
    convOutput->Add(biasHor->elem[0]);
    ConvoluteStandard2D2DFullySymmetric(vertOutput, convOutput, kernelHor);
}


void BottleneckConvolutionStandardLimited(tensor* inputStart, tensor* inputLast, matrix* vertOutput, matrix* convOutput, tensor* kernelVertStart,
                                          tensor* kernelVertLast, matrix* kernelHor, realNumber* biasHor){
    ConvolutionStandardVertical1D(inputStart, kernelVertStart, vertOutput);
    AddConvolutionStandardVertical1D(inputLast, kernelVertLast, vertOutput);
    convOutput->Add(biasHor->elem[0]);
    ConvoluteStandard2D2D(vertOutput, convOutput, kernelHor);
}

void BackwardBottleneckConvolutionStandard(tensor* input, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernel_vert, matrix* kernel_hor, tensor* kernelGrad_vert, matrix* kernelGrad_hor, realNumber* biasGrad_hor){
    BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernel_hor, kernelGrad_hor, biasGrad_hor);
    BackwardConvolutionStandardVertical1D(input, inputDelta, vertOutputDelta, kernel_vert, kernelGrad_vert);
}

void BackwardBottleneckConvolutionStandardRandom(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                                tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, realNumber* biasGradHor){
    BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, biasGradHor);
    BackwardConvolutionStandardVerticalRandom1D(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert);
}

void BackwardBottleneckConvolutionStandardRandomSymmetric(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                                tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, realNumber* biasGradHor){
    BackwardConvoluteStandard2D2DSymmetric(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, biasGradHor);
    BackwardConvolutionStandardVerticalRandom1D(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert);
}

void BackwardBottleneckConvolutionStandardRandomSymmetricDrop(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput,
                                                    matrix* vertOutputDelta, tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor,
                                                    realNumber* biasGradHor, activityData* vertOutputActivity){
    BackwardConvoluteStandard2D2DSymmetric(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, biasGradHor);
    vertOutputDelta->SetDroppedElementsToZero(vertOutputActivity);
    BackwardConvolutionStandardVerticalRandom1D(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert);
}

void BackwardBottleneckConvolutionStandardRandomSymmetricNoBiasDrop(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput,
                                                    matrix* vertOutputDelta, tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor,
                                                    activityData* vertOutputActivity){
    BackwardConvoluteStandard2D2DSymmetric(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, NULL);
    vertOutputDelta->SetDroppedElementsToZero(vertOutputActivity);
    BackwardConvolutionStandardVerticalRandom1D(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert);
}

void BackwardSymmetricConvolution(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput,
                                                    matrix* vertOutputDelta, tensor* kernelVert, matrix* kernelHor,
                                                    tensor* kernelGradVert, matrix* kernelGradHor, int symmetryLevel){

    if (symmetryLevel == 0)
        BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, NULL);
    if (symmetryLevel == 1)
        BackwardConvoluteStandard2D2DSymmetric(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, NULL);
    if (symmetryLevel == 2)
        BackwardConvoluteStandard2D2DSymmetric2(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, NULL);
    if (symmetryLevel == 3)
        BackwardConvoluteStandard2D2DSymmetric3(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, NULL);
    if (symmetryLevel == 4)
        BackwardConvoluteStandard2D2DFullySymmetric(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, NULL);

    BackwardConvolutionStandardVerticalRandom1D(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert);
}


void BackwardPyramidalConvolution(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                  tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, int stair){
    BackwardPyramidalConvolution2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, stair);
    BackwardPyramidalVerticalConvolution(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert, stair);
}



void BackwardBottleneckConvolutionStandardRandomFullySymmetric(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                                tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, realNumber* biasGradHor){
    BackwardConvoluteStandard2D2DFullySymmetric(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, biasGradHor);
    BackwardConvolutionStandardVerticalRandom1D(input, indices, inputDelta, vertOutputDelta, kernelVert, kernelGradVert);
}

void BackwardBottleneckConvolutionStandardLimited(tensor* inputStart, tensor* inputLast, tensor* inputLastDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernelVertLast, matrix* kernelHor, tensor* kernelGradVertStart, tensor* kernelGradVertLast, matrix* kernelGradHor, realNumber* biasGradHor){
    BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, biasGradHor);
    BackwardConvolutionStandardVertical1D(inputLast, inputLastDelta, vertOutputDelta, kernelVertLast, kernelGradVertLast);
    BackwardConvolutionStandardVertical1DGrad(inputStart, vertOutputDelta, kernelGradVertStart);
}

void BackwardBottleneckConvolutionStandardLimitedFull(tensor* inputStart, tensor* inputLast, tensor* inputStartDelta, tensor* inputLastDelta,
                            matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta, tensor* kernelVertStart,
                            tensor* kernelVertLast, matrix* kernelHor, tensor* kernelGradVertStart, tensor* kernelGradVertLast, matrix* kernelGradHor, realNumber* biasGradHor){
    BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernelHor, kernelGradHor, biasGradHor);
    BackwardConvolutionStandardVertical1D(inputLast, inputLastDelta, vertOutputDelta, kernelVertLast, kernelGradVertLast);
    BackwardConvolutionStandardVertical1D(inputStart, inputStartDelta, vertOutputDelta, kernelVertStart, kernelGradVertStart);
}

void BackwardBottleneckConvolutionStandardPartialGrad(tensor* input, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernel_vert, matrix* kernel_hor, tensor* kernelGrad_vert, matrix* kernelGrad_hor, realNumber* biasGrad_hor, int startDepth){
    BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernel_hor, kernelGrad_hor, biasGrad_hor);
    BackwardConvolutionStandardVertical1DPartialGrad(input, inputDelta, vertOutputDelta, kernel_vert, kernelGrad_vert, startDepth);
}


void BackwardBottleneckConvolutionStandardGrad(tensor* input, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernel_vert, matrix* kernel_hor, tensor* kernelGrad_vert, matrix* kernelGrad_hor, realNumber* biasGrad_hor){
    BackwardConvoluteStandard2D2D(vertOutput, vertOutputDelta, convOutputDelta, kernel_hor, kernelGrad_hor, biasGrad_hor);
    BackwardConvolutionStandardVertical1DGrad(input, vertOutputDelta, kernelGrad_vert);
}






void Convolute3D2D(tensor* input, matrix* output, tensor* reversedKernel, realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow){
    matrix* input_d = new matrix();
    matrix* reversedKernel_d = new matrix();

    for(int d=0; d<reversedKernel->depth; d++){
        input_d->SetToTensorLayer(input, d);
        reversedKernel_d->SetToTensorLayer(reversedKernel, d);

        if (d==0)
            Convolute2D2D(input_d, output, reversedKernel_d, bias, paddingR, paddingC, indexInputRow);
        else
            Convolute2D2D(input_d, output, reversedKernel_d, NULL, paddingR, paddingC, indexInputRow);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(reversedKernel_d);
}

void ConvoluteStandard3D2D(tensor* input, matrix* output, tensor* kernel, realNumber* bias){
    if (input->rows == 2){
        ConvoluteStandard3D2D_2_2(input, output, kernel, bias);
        return;
    }

    output->Add(bias->elem[0]);
    const int input_rows = input->rows, input_cols = input->cols;
    double * output_oR, * input_d, * kernel_d;
    for(int d=0; d<kernel->depth; d++){
        input_d = input->Layer(d);
        kernel_d = kernel->Layer(d);
        output_oR = output->Row(0);
        ConvoluteStandard(input_d, output_oR, input_cols, kernel_d+3);
        ConvoluteStandard(input_d + input_cols, output_oR, input_cols, kernel_d+6);

        for(int oR=1; oR<input_rows-1; ++oR){
            output_oR = output->Row(oR);
            ConvoluteStandard(input_d + input_cols * (oR-1), output_oR, input_cols, kernel_d);
            ConvoluteStandard(input_d + input_cols * oR    , output_oR, input_cols, kernel_d+3);
            ConvoluteStandard(input_d + input_cols * (oR+1), output_oR, input_cols, kernel_d+6);
        }

        output_oR = output->Row(input_rows-1);
        ConvoluteStandard(input_d+input_cols*(input_rows-2), output_oR, input_cols, kernel_d);
        ConvoluteStandard(input_d+input_cols*(input_rows-1), output_oR, input_cols, kernel_d+3);
    }

//
//    matrix* input_d = new matrix();
//    matrix* kernel_d = new matrix();
//
//    for(int d=0; d<kernel->depth; d++){
//        input_d->SetToTensorLayer(input, d);
//        kernel_d->SetToTensorLayer(kernel, d);
//        ConvoluteStandard2D2D(input_d, output, kernel_d);
//    }
//
//    DeleteOnlyShell(input_d);
//    DeleteOnlyShell(kernel_d);
}


void ConvoluteMultipleStandard3D3D(tensor* input, tensor* output, tensor* kernel, vect* bias){
    matrix* input_d = new matrix();
    matrix* kernel_d = new matrix();
    matrix* output_d = new matrix();
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        kernel_d->SetToTensorLayer(kernel, d);
        output_d->SetToTensorLayer(output, d);

        output_d->Add(bias->elem[d]);
        ConvoluteStandard2D2D(input_d, output_d, kernel_d);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(kernel_d);
    DeleteOnlyShell(output_d);
}

void BackwardConvoluteMultipleStandard3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad){
    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();
    matrix* outputDelta_d = new matrix();
    matrix* kernel_d = new matrix();
    matrix* kernelGrad_d = new matrix();
    realNumber* biasGrad_d = new realNumber();

    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        outputDelta_d->SetToTensorLayer(outputDelta, d);
        kernel_d->SetToTensorLayer(kernel, d);
        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
        biasGrad_d->SetToVectElement(biasGrad, d);

        BackwardConvoluteStandard2D2D(input_d, inputDelta_d, outputDelta_d, kernel_d, kernelGrad_d, biasGrad_d);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);
    DeleteOnlyShell(outputDelta_d);
    DeleteOnlyShell(kernel_d);
    DeleteOnlyShell(kernelGrad_d);
    DeleteOnlyShell(biasGrad_d);
}




void BackwardConvolute3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();
    matrix* kernel_d = new matrix();
    matrix* kernelGrad_d = new matrix();

    for(int d=0; d<kernel->depth; d++){
        input_d->SetToTensorLayer(input, d);
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        kernel_d->SetToTensorLayer(kernel, d);
        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
        if (d==0)
            BackwardConvolute2D2D(input_d, inputDelta_d, outputDelta, kernel_d, kernelGrad_d, biasGrad,      paddingR, paddingC, indexOutputRow);
        else
            BackwardConvolute2D2D(input_d, inputDelta_d, outputDelta, kernel_d, kernelGrad_d, NULL, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);
    DeleteOnlyShell(kernel_d);
    DeleteOnlyShell(kernelGrad_d);
}


void BackwardConvoluteStandard3D2D_4_4(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, realNumber* biasGrad){
    biasGrad->elem[0]+=outputDelta->Sum();

    double* inputDelta_iR;
    double* input_iR;
    double * input_d, * inputDelta_d, * kernel_d, * kernelGrad_d;

    for(int d=0; d<kernel->depth; d++){
        input_d = input->Layer(d);
        inputDelta_d = inputDelta->Layer(d);
        kernel_d = kernel->Layer(d);
        kernelGrad_d = kernelGrad->Layer(d);

        input_iR = input_d;
        inputDelta_iR = inputDelta_d;
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(0), kernel_d+3, kernelGrad_d+3);
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(1), kernel_d, kernelGrad_d);
        for(int iR=1; iR<3; ++iR){
            input_iR = input_d + 4 * iR;
            inputDelta_iR = inputDelta_d + 4 * iR;
            BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel_d+6, kernelGrad_d+6);
            BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel_d+3, kernelGrad_d+3);
            BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel_d  , kernelGrad_d);
        }
        input_iR = input_d + 12;
        inputDelta_iR = inputDelta_d + 12;
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(2), kernel_d+6, kernelGrad_d+6);
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(3), kernel_d+3, kernelGrad_d+3);
    }
}

void BackwardConvoluteStandard2D2D_4_4(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad){
    double* inputDelta_iR;
    double* input_iR;
    if (biasGrad)
        biasGrad->elem[0]+=outputDelta->Sum();

    input_iR = input->elem;
    inputDelta_iR = inputDelta->elem;

    BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(0), kernel->elem+3, kernelGrad->elem+3);
    BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(1), kernel->elem, kernelGrad->elem);
    for(int iR=1; iR<3; ++iR){
        input_iR = input->elem + 4 * iR;
        inputDelta_iR = inputDelta->elem + 4 * iR;
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel->elem+6, kernelGrad->elem+6);
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel->elem+3, kernelGrad->elem+3);
        BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel->elem  , kernelGrad->elem);
    }
    input_iR = input->elem + 12;
    inputDelta_iR = inputDelta->elem + 12;
    BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(2), kernel->elem+6, kernelGrad->elem+6);
    BackwardConvoluteStandard_4(input_iR, inputDelta_iR, outputDelta->Row(3), kernel->elem+3, kernelGrad->elem+3);
}

void BackwardConvoluteStandard3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, realNumber* biasGrad){

    if (input->rows == 2){
        BackwardConvoluteStandard3D2D_2_2(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad);
        return;
    }

    if (input->rows == 4){
        BackwardConvoluteStandard3D2D_4_4(input, inputDelta, outputDelta, kernel, kernelGrad, biasGrad);
        return;
    }

    biasGrad->elem[0]+=outputDelta->Sum();

    double* inputDelta_iR;
    double* input_iR;
    double * input_d, * inputDelta_d, * kernel_d, * kernelGrad_d;
    int input_rows = input->rows, input_cols = input->cols;

    for(int d=0; d<kernel->depth; d++){
        input_d = input->Layer(d);
        inputDelta_d = inputDelta->Layer(d);
        kernel_d = kernel->Layer(d);
        kernelGrad_d = kernelGrad->Layer(d);

        input_iR = input_d;
        inputDelta_iR = inputDelta_d;
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(0), kernel_d+3, kernelGrad_d+3, input_cols);
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(1), kernel_d  , kernelGrad_d, input_cols);
        for(int iR=1; iR<input_rows-1; ++iR){
            input_iR = input_d + input_cols*iR;
            inputDelta_iR = inputDelta_d + input_cols*iR;
            BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(iR-1), kernel_d+6, kernelGrad_d+6, input_cols);
            BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(iR),   kernel_d+3, kernelGrad_d+3, input_cols);
            BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(iR+1), kernel_d, kernelGrad_d, input_cols);
        }
        input_iR = input_d + input_cols*(input_rows-1);
        inputDelta_iR = inputDelta_d + input_cols*(input_rows-1);
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(input_rows-2), kernel_d+6, kernelGrad_d+6, input_cols);
        BackwardConvoluteStandard(input_iR, inputDelta_iR, outputDelta->Row(input_rows-1), kernel_d+3, kernelGrad_d+3, input_cols);
    }
//
//
//    matrix* input_d = new matrix();
//    matrix* inputDelta_d = new matrix();
//    matrix* kernel_d = new matrix();
//    matrix* kernelGrad_d = new matrix();
//
//    for(int d=0; d<kernel->depth; d++){
//        input_d->SetToTensorLayer(input, d);
//        inputDelta_d->SetToTensorLayer(inputDelta, d);
//        kernel_d->SetToTensorLayer(kernel, d);
//        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
//        if (d==0)
//            BackwardConvoluteStandard2D2D(input_d, inputDelta_d, outputDelta, kernel_d, kernelGrad_d, biasGrad);
//        else
//            BackwardConvoluteStandard2D2D(input_d, inputDelta_d, outputDelta, kernel_d, kernelGrad_d, NULL);
//    }
//
//    DeleteOnlyShell(input_d);
//    DeleteOnlyShell(inputDelta_d);
//    DeleteOnlyShell(kernel_d);
//    DeleteOnlyShell(kernelGrad_d);
}

void BackwardConvoluteGradStandard3D2D_4_4(tensor* input, matrix* outputDelta, tensor* kernelGrad, realNumber* biasGrad){
    biasGrad->elem[0]+=outputDelta->Sum();

    double* input_iR;
    double * input_d, * kernelGrad_d;

    for(int d=0; d<kernelGrad->depth; d++){
        input_d = input->Layer(d);
        kernelGrad_d = kernelGrad->Layer(d);

        input_iR = input_d;
        BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(0), kernelGrad_d+3);
        BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(1), kernelGrad_d);
        for(int iR=1; iR<3; ++iR){
            input_iR = input_d + 4*iR;
            BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(iR-1), kernelGrad_d+6);
            BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(iR),   kernelGrad_d+3);
            BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(iR+1), kernelGrad_d);
        }
        input_iR = input_d + 12;
        BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(2), kernelGrad_d+6);
        BackwardConvoluteGradStandard_4(input_iR, outputDelta->Row(3), kernelGrad_d+3);
    }
}

void BackwardConvoluteGradStandard3D2D(tensor* input, matrix* outputDelta, tensor* kernelGrad, realNumber* biasGrad){

    if (input->rows == 2){
        BackwardConvoluteGradStandard3D2D_2_2(input, outputDelta, kernelGrad, biasGrad);
        return;
    }

    if (input->rows == 4){
        BackwardConvoluteGradStandard3D2D_4_4(input, outputDelta, kernelGrad, biasGrad);
        return;
    }

    biasGrad->elem[0]+=outputDelta->Sum();

    double* input_iR;
    double * input_d, * kernelGrad_d;
    int input_rows = input->rows, input_cols = input->cols;

    for(int d=0; d<kernelGrad->depth; d++){
        input_d = input->Layer(d);
        kernelGrad_d = kernelGrad->Layer(d);

        input_iR = input_d;
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(0), kernelGrad_d+3, input_cols);
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(1), kernelGrad_d, input_cols);
        for(int iR=1; iR<input_rows-1; ++iR){
            input_iR = input_d + input_cols*iR;
            BackwardConvoluteGradStandard(input_iR, outputDelta->Row(iR-1), kernelGrad_d+6, input_cols);
            BackwardConvoluteGradStandard(input_iR, outputDelta->Row(iR),   kernelGrad_d+3, input_cols);
            BackwardConvoluteGradStandard(input_iR, outputDelta->Row(iR+1), kernelGrad_d  , input_cols);
        }
        input_iR = input_d + input_cols*(input_rows-1);
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(input_rows-2), kernelGrad_d+6, input_cols);
        BackwardConvoluteGradStandard(input_iR, outputDelta->Row(input_rows-1), kernelGrad_d+3, input_cols);
    }
//
//    if (input->rows == 2){
//        BackwardConvoluteGradStandard3D2D_2_2(input, outputDelta, kernelGrad, biasGrad);
//        return;
//    }
//    matrix* input_d = new matrix();
//    matrix* kernelGrad_d = new matrix();
//
//    for(int d=0; d<kernelGrad->depth; d++){
//        input_d->SetToTensorLayer(input, d);
//        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
//        if (d==0)
//            BackwardConvoluteGradStandard2D2D(input_d, outputDelta, kernelGrad_d, biasGrad);
//        else
//            BackwardConvoluteGradStandard2D2D(input_d, outputDelta, kernelGrad_d, NULL);
//    }
//
//    DeleteOnlyShell(input_d);
//    DeleteOnlyShell(kernelGrad_d);
}



void BackwardConvoluteGrad3D2D(tensor* input, matrix* outputDelta, tensor* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow){

    matrix* input_d = new matrix();
    matrix* kernelGrad_d = new matrix();

    for(int d=0; d<kernelGrad->depth; d++){
        input_d->SetToTensorLayer(input, d);
        kernelGrad_d->SetToTensorLayer(kernelGrad, d);
        if (d==0)
            BackwardConvoluteGrad2D2D(input_d, outputDelta, kernelGrad_d, biasGrad,      paddingR, paddingC, indexOutputRow);
        else
            BackwardConvoluteGrad2D2D(input_d, outputDelta, kernelGrad_d, NULL, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(kernelGrad_d);
}





void ConvoluteQuadratic3D2D(tensor* input, matrix* output, tensor* linearReversedKernel, tensor* quadraticReversedKernel,
                             realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow){
    matrix* input_d = new matrix();

    matrix* linearReversedKernel_d = new matrix();
    matrix* quadraticReversedKernel_d = new matrix();

    for(int d=0; d<linearReversedKernel->depth; d++){
        input_d->SetToTensorLayer(input, d);

        linearReversedKernel_d->SetToTensorLayer(linearReversedKernel, d);
        quadraticReversedKernel_d->SetToTensorLayer(quadraticReversedKernel, d);

        if (d==0)
            ConvoluteQuadratic2D2D(input_d, output, linearReversedKernel_d, quadraticReversedKernel_d, bias, paddingR, paddingC, indexInputRow);
        else
            ConvoluteQuadratic2D2D(input_d, output, linearReversedKernel_d, quadraticReversedKernel_d, NULL, paddingR, paddingC, indexInputRow);
    }

    DeleteOnlyShell(input_d);

    DeleteOnlyShell(linearReversedKernel_d);
    DeleteOnlyShell(quadraticReversedKernel_d);
}


void BackwardConvoluteQuadratic3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* linearKernel, tensor* quadraticKernel,
                                    tensor* linearKernelGrad, tensor* quadraticKernelGrad, realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){
    matrix* input_d = new matrix();
    matrix* inputDelta_d = new matrix();

    matrix* linearKernel_d = new matrix();
    matrix* quadraticKernel_d = new matrix();

    matrix* linearKernelGrad_d = new matrix();
    matrix* quadraticKernelGrad_d = new matrix();

    for(int d=0; d<linearKernel->depth; d++){
        input_d->SetToTensorLayer(input, d);
        inputDelta_d->SetToTensorLayer(inputDelta, d);

        linearKernel_d->SetToTensorLayer(linearKernel, d);
        quadraticKernel_d->SetToTensorLayer(quadraticKernel, d);

        linearKernelGrad_d->SetToTensorLayer(linearKernelGrad, d);
        quadraticKernelGrad_d->SetToTensorLayer(quadraticKernelGrad, d);

        if (d==0)
            BackwardConvoluteQuadratic2D2D(input_d, inputDelta_d, outputDelta, linearKernel_d, quadraticKernel_d,
                                           linearKernelGrad_d, quadraticKernelGrad_d, biasGrad  , paddingR, paddingC, indexOutputRow);
        else
            BackwardConvoluteQuadratic2D2D(input_d, inputDelta_d, outputDelta, linearKernel_d, quadraticKernel_d,
                                           linearKernelGrad_d, quadraticKernelGrad_d, NULL      , paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(inputDelta_d);

    DeleteOnlyShell(linearKernel_d);
    DeleteOnlyShell(quadraticKernel_d);

    DeleteOnlyShell(linearKernelGrad_d);
    DeleteOnlyShell(quadraticKernelGrad_d);
}


void BackwardConvoluteQuadraticGrad3D2D(tensor* input, matrix* outputDelta, tensor* linearKernelGrad, tensor* quadraticKernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow){

    matrix* input_d = new matrix();

    matrix* linearKernelGrad_d = new matrix();
    matrix* quadraticKernelGrad_d = new matrix();

    for(int d=0; d<linearKernelGrad->depth; d++){
        input_d->SetToTensorLayer(input, d);

        linearKernelGrad_d->SetToTensorLayer(linearKernelGrad, d);
        quadraticKernelGrad_d->SetToTensorLayer(quadraticKernelGrad, d);

        if (d==0)
            BackwardConvoluteQuadraticGrad2D2D(input_d, outputDelta, linearKernelGrad_d, quadraticKernelGrad_d, biasGrad,    paddingR, paddingC, indexOutputRow);
        else
            BackwardConvoluteQuadraticGrad2D2D(input_d, outputDelta, linearKernelGrad_d, quadraticKernelGrad_d, NULL,        paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(input_d);

    DeleteOnlyShell(linearKernelGrad_d);
    DeleteOnlyShell(quadraticKernelGrad_d);
}





void Convolute3D3D(tensor* input, tensor* output, tensor4D* reversedKernel, vect* bias, int paddingR, int paddingC, vector<int> & indexInputRow){
    matrix* output_n = new matrix();
    realNumber* bias_n = new realNumber();
    tensor* reversedKernel_n = new tensor();

    for(int n=0; n<reversedKernel->number; ++n){
        output_n->SetToTensorLayer(output, n);
        reversedKernel_n->SetToTLayer(reversedKernel, n);
        bias_n->SetToVectElement(bias, n);

        Convolute3D2D(input, output_n, reversedKernel_n, bias_n, paddingR, paddingC, indexInputRow);
    }

    DeleteOnlyShell(output_n);
    DeleteOnlyShell(reversedKernel_n);
    DeleteOnlyShell(bias_n);
}

void BackwardConvolute3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

    matrix* outputDelta_n = new matrix();
    tensor* kernel_n = new tensor();
    tensor* kernelGrad_n = new tensor();
    realNumber* biasGrad_n = new realNumber();

    for(int n=0; n<kernel->number; ++n){
        outputDelta_n->SetToTensorLayer(outputDelta, n);
        kernel_n->SetToTLayer(kernel, n);
        kernelGrad_n->SetToTLayer(kernelGrad, n);
        biasGrad_n->SetToVectElement(biasGrad, n);

        BackwardConvolute3D2D(input, inputDelta, outputDelta_n, kernel_n, kernelGrad_n, biasGrad_n, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(outputDelta_n);
    DeleteOnlyShell(kernel_n);
    DeleteOnlyShell(kernelGrad_n);
    DeleteOnlyShell(biasGrad_n);
}

void BackwardConvoluteGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){

    matrix* outputDelta_n = new matrix();
    tensor* kernelGrad_n = new tensor();
    realNumber* biasGrad_n = new realNumber();

    for(int n=0; n<kernelGrad->number; ++n){
        outputDelta_n->SetToTensorLayer(outputDelta, n);
        kernelGrad_n->SetToTLayer(kernelGrad, n);
        biasGrad_n->SetToVectElement(biasGrad, n);
        BackwardConvoluteGrad3D2D(input, outputDelta_n, kernelGrad_n, biasGrad_n, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(outputDelta_n);
    DeleteOnlyShell(kernelGrad_n);
    DeleteOnlyShell(biasGrad_n);
}





void ConvoluteQuadratic3D3D(tensor* input, tensor* output, tensor4D* linearReversedKernel, tensor4D* quadraticReversedKernel,
                            vect* bias, int paddingR, int paddingC, vector<int> & indexInputRow){
    matrix* output_n = new matrix();
    realNumber* bias_n = new realNumber();
    tensor* linearReversedKernel_n = new tensor();
    tensor* quadraticReversedKernel_n = new tensor();

    for(int n=0; n<linearReversedKernel->number; ++n){
        output_n->SetToTensorLayer(output, n);
        linearReversedKernel_n->SetToTLayer(linearReversedKernel, n);
        quadraticReversedKernel_n->SetToTLayer(quadraticReversedKernel, n);
        bias_n->SetToVectElement(bias, n);

        ConvoluteQuadratic3D2D(input, output_n, linearReversedKernel_n, quadraticReversedKernel_n, bias_n, paddingR, paddingC, indexInputRow);
    }

    DeleteOnlyShell(output_n);
    DeleteOnlyShell(linearReversedKernel_n);
    DeleteOnlyShell(quadraticReversedKernel_n);
    DeleteOnlyShell(bias_n);
}


void BackwardConvoluteQuadratic3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* linearKernel, tensor4D* quadraticKernel,
                                       tensor4D* linearKernelGrad, tensor4D* quadraticKernelGrad, vect* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow){
    matrix* outputDelta_n = new matrix();
    tensor* linearKernel_n = new tensor();
    tensor* linearKernelGrad_n = new tensor();
    tensor* quadraticKernel_n = new tensor();
    tensor* quadraticKernelGrad_n = new tensor();
    realNumber* biasGrad_n = new realNumber();

    for(int n=0; n<linearKernel->number; ++n){
        outputDelta_n->SetToTensorLayer(outputDelta, n);
        linearKernel_n->SetToTLayer(linearKernel, n);
        linearKernelGrad_n->SetToTLayer(linearKernelGrad, n);
        quadraticKernel_n->SetToTLayer(quadraticKernel, n);
        quadraticKernelGrad_n->SetToTLayer(quadraticKernelGrad, n);
        biasGrad_n->SetToVectElement(biasGrad, n);

        BackwardConvoluteQuadratic3D2D(input, inputDelta, outputDelta_n, linearKernel_n, quadraticKernel_n,
                                       linearKernelGrad_n, quadraticKernelGrad_n, biasGrad_n, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(outputDelta_n);
    DeleteOnlyShell(linearKernel_n);
    DeleteOnlyShell(linearKernelGrad_n);
    DeleteOnlyShell(quadraticKernel_n);
    DeleteOnlyShell(quadraticKernelGrad_n);
    DeleteOnlyShell(biasGrad_n);
}


void BackwardConvoluteQuadraticGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* linearKernelGrad, tensor4D* quadraticKernelGrad,
                                        vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow){
    matrix* outputDelta_n = new matrix();
    tensor* linearKernelGrad_n = new tensor();
    tensor* quadraticKernelGrad_n = new tensor();
    realNumber* biasGrad_n = new realNumber();

    for(int n=0; n<linearKernelGrad->number; ++n){
        outputDelta_n->SetToTensorLayer(outputDelta, n);
        linearKernelGrad_n->SetToTLayer(linearKernelGrad, n);
        quadraticKernelGrad_n->SetToTLayer(quadraticKernelGrad, n);
        biasGrad_n->SetToVectElement(biasGrad, n);
        BackwardConvoluteQuadraticGrad3D2D(input, outputDelta_n, linearKernelGrad_n, quadraticKernelGrad_n, biasGrad_n, paddingR, paddingC, indexOutputRow);
    }

    DeleteOnlyShell(outputDelta_n);
    DeleteOnlyShell(linearKernelGrad_n);
    DeleteOnlyShell(quadraticKernelGrad_n);
    DeleteOnlyShell(biasGrad_n);
}


void SequentialConvolutionStandard(tensor* input, tensor* kernel, vect* bias, int startDepth, activityData* inputActivity){
    int convolutionNumber = bias->len;

    tensor* input_j = new tensor();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    tensor* kernel_j = new tensor();
    realNumber* bias_j = new realNumber();

    for(int j=0; j<convolutionNumber; ++j){
        input_j->SubTensor(input, startDepth + 2 * j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernel_j->SubTensor(kernel, j * (startDepth + j - 1), startDepth + 2 * j);
        bias_j->SetToVectElement(bias, j);

        ConvoluteStandard3D2D(input_j, convOutput_j, kernel_j, bias_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        //minOutput_j->SetToMinReluFunction(convOutput_j);
        //convOutput_j->SetToReluFunction();
        input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * input->rows * input->cols);
    }

    DeleteOnlyShell(input_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(kernel_j);
    DeleteOnlyShell(bias_j);
}

void SequentialMaxMinStandard(tensor* input, vect* kernel, vect* bias, int startDepth, int nConvolutions, activityData* inputActivity, bool testMode){
    tensor* input_j = new tensor();
    vect* kernel_j = new vect();
    double bias_j;
    for(int j=0; j<nConvolutions; ++j){
        input_j->SubTensor(input, startDepth + 2 * j);
        kernel_j->SubVect(kernel, j * (startDepth + j - 1), startDepth + 2 * j);
        bias_j = bias->elem[j];
        input->elem[startDepth + 2 * j] = InnerProduct(input_j, kernel_j) + bias_j;
        input->elem[startDepth + 2 * j + 1] = min(input->elem[startDepth + 2 * j], 0.0);
        input->elem[startDepth + 2 * j]     = max(input->elem[startDepth + 2 * j], 0.0);
        if (!testMode && inputActivity->dropping){
            input->elem[startDepth + 2 * j]     *= inputActivity->activeUnits[startDepth + 2 * j];
            input->elem[startDepth + 2 * j + 1] *= inputActivity->activeUnits[startDepth + 2 * j + 1];
        }
    }
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(kernel_j);
}


void SequentialBottleneckConvolutionReluStandard(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                                             tensor* verticalConv, int startDepth, int nConvolutions, activityData* inputActivity){
    //verticalConv->SetToZero();

    tensor* input_j = new tensor();
    matrix* convOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernel_vert_j = new tensor();
    matrix* kernel_hor_j = new matrix();
    realNumber* bias_hor_j = new realNumber();

    for(int j=0; j<nConvolutions; ++j){
        input_j->SubTensor(input, startDepth + j);
        convOutput_j->SetToTensorLayer(input, startDepth + j);
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        kernel_vert_j->SubTensor(kernel_vert, j * (2 * startDepth + j - 1) / 2, startDepth + j);
        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
        bias_hor_j->SetToVectElement(bias_hor, j);
        BottleneckConvolutionStandard(input_j, vertOutput_j, convOutput_j, kernel_vert_j, kernel_hor_j, bias_hor_j);
        convOutput_j->SetToReluFunction();
        input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + j), convOutput_j->len);
    }
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernel_vert_j);
    DeleteOnlyShell(kernel_hor_j);
    DeleteOnlyShell(bias_hor_j);
}


void SequentialBottleneckConvolutionStandard(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                                             tensor* verticalConv, int startDepth, int nConvolutions, activityData* inputActivity, bool testMode){
    //verticalConv->SetToZero();

    tensor* input_j = new tensor();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernel_vert_j = new tensor();
    matrix* kernel_hor_j = new matrix();
    realNumber* bias_hor_j = new realNumber();

    for(int j=0; j<nConvolutions; ++j){
        input_j->SubTensor(input, startDepth + 2 * j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        kernel_vert_j->SubTensor(kernel_vert, j * (startDepth + j - 1), startDepth + 2 * j);
        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
        bias_hor_j->SetToVectElement(bias_hor, j);
        BottleneckConvolutionStandard(input_j, vertOutput_j, convOutput_j, kernel_vert_j, kernel_hor_j, bias_hor_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * input->rows * input->cols);
    }
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernel_vert_j);
    DeleteOnlyShell(kernel_hor_j);
    DeleteOnlyShell(bias_hor_j);
}



void SequentialBottleneckConvolutionStandardLimited(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                                             tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, bool testMode){
    //verticalConv->SetToZero();
    tensor* inputStart = new tensor();
    tensor* inputLast_j = new tensor();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVertStart_j = new tensor();
    tensor* kernelVertLast_j = new tensor();
    matrix* kernelHor_j = new matrix();
    realNumber* biasHor_j = new realNumber();

    inputStart->SubTensor(input, alwaysPresentDepth);

    int vertKernelStartIndex = 0;

    for(int j=0; j<nConvolutions; ++j){
        inputLast_j->SubTensor(input, startDepth + 2 * j - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth), min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        kernelVertStart_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth);
        kernelVertLast_j->SubTensor(kernel_vert,  vertKernelStartIndex + alwaysPresentDepth, min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        biasHor_j->SetToVectElement(bias_hor, j);
        BottleneckConvolutionStandardLimited(inputStart, inputLast_j, vertOutput_j, convOutput_j, kernelVertStart_j, kernelVertLast_j, kernelHor_j, biasHor_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
    }

    DeleteOnlyShell(inputStart);
    DeleteOnlyShell(inputLast_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVertStart_j);
    DeleteOnlyShell(kernelVertLast_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(biasHor_j);
}




void SequentialBottleneckConvolutionStandardRandom(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices, bool testMode){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();
    realNumber* biasHor_j = new realNumber();

    int vertKernelStartIndex = 0;
    int * indices_j;

    for(int j=0; j<nConvolutions; ++j){
        indices_j = indices + vertKernelStartIndex;
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        biasHor_j->SetToVectElement(bias_hor, j);

        BottleneckConvolutionStandardRandom(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, biasHor_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
    }
    if (vertKernelStartIndex != kernel_vert->len)
        cout<<"Error in forward random pass"<<endl;

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(biasHor_j);
}


void SequentialBottleneckConvolutionStandardRandomSymmetric(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity,
                activityData* verticalConvActivity, int* indices, bool testMode){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();
    realNumber* biasHor_j = new realNumber();
    activityData* verticalConvActivity_j = new activityData();

    int vertKernelStartIndex = 0;
    int * indices_j;

    for(int j=0; j<nConvolutions; ++j){
        indices_j = indices + vertKernelStartIndex;
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        biasHor_j->SetToVectElement(bias_hor, j);
        verticalConvActivity_j->SubActivityData(verticalConvActivity, j * vertOutput_j->len, vertOutput_j->len);

        BottleneckConvolutionStandardRandomSymmetricDrop(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, biasHor_j, verticalConvActivity_j, testMode);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
    }
    if (vertKernelStartIndex != kernel_vert->len)
        cout<<"Error in forward random pass"<<endl;

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(biasHor_j);
    DeleteOnlyShellActivity(verticalConvActivity_j);
}




void SequentialBottleneckConvolutionStandardRandomSymmetricNoBias(tensor* input, tensor* kernel_vert, tensor* kernel_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity,
                activityData* verticalConvActivity, int* indices, bool testMode){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();
    activityData* verticalConvActivity_j = new activityData();

    int vertKernelStartIndex = 0;
    int * indices_j;

    for(int j=0; j<nConvolutions; ++j){
        indices_j = indices + vertKernelStartIndex;
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        verticalConvActivity_j->SubActivityData(verticalConvActivity, j * vertOutput_j->len, vertOutput_j->len);

        BottleneckConvolutionStandardRandomSymmetricNoBiasDrop(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, verticalConvActivity_j, testMode);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
    }
    if (vertKernelStartIndex != kernel_vert->len)
        cout<<"Error in forward random pass"<<endl;

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShellActivity(verticalConvActivity_j);
}





void ForwardStairsConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, activityData* verticalConvActivity, int* indices, bool testMode){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();
    activityData* verticalConvActivity_j = new activityData();

    int vertKernelStartIndex = 0;
    int * indices_j;
    int j=0;
    for(int stair=0; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            j = stair * numStairConvolutions + conv;
            indices_j = indices + vertKernelStartIndex;
            vertOutput_j->SetToTensorLayer(verticalConv, j);
            convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
            minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
            kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, startDepth + 2 * numStairConvolutions * stair);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            verticalConvActivity_j->SubActivityData(verticalConvActivity, j * vertOutput_j->len, vertOutput_j->len);
            BottleneckConvolutionStandardRandomSymmetricNoBiasDrop(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, verticalConvActivity_j, testMode);
            minOutput_j->BranchMaxMin(convOutput_j);
            if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
            vertKernelStartIndex += startDepth + 2 * numStairConvolutions * stair;
        }

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShellActivity(verticalConvActivity_j);
}



void ForwardStairsSymmetricConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, int* indices, bool testMode, int symmetryLevel){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    int vertKernelStartIndex = 0;
    int * indices_j;
    int j=0;
    for(int stair=0; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            j = stair * numStairConvolutions + conv;
            indices_j = indices + vertKernelStartIndex;
            vertOutput_j->SetToTensorLayer(verticalConv, j);
            convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
            minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
            kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, startDepth + 2 * numStairConvolutions * stair);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            SymmetricConvolution(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, symmetryLevel);
            minOutput_j->BranchMaxMin(convOutput_j);
            if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
            vertKernelStartIndex += startDepth + 2 * numStairConvolutions * stair;
        }

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
}




void ForwardStairsSymmetricConvolutionRelu(tensor* input, tensor* kernel, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, bool testMode, int symmetryLevel){


    tensor* input_stair = new tensor();
    tensor* output_stair = new tensor();
    tensor* kernel_stair = new tensor();
    vect* bias_stair = new vect(0);
    int kernelStartDepth = 0;
    for(int stair=0; stair<numStairs; ++stair){
        input_stair->SubTensor(input, startDepth + stair * numStairConvolutions);
        output_stair->SubTensor(input, startDepth + stair * numStairConvolutions, numStairConvolutions);
        kernel_stair->SubTensor(kernel, kernelStartDepth, (startDepth + stair * numStairConvolutions) * numStairConvolutions);
        SymmetricConvolution3D(input_stair, output_stair, kernel_stair, bias_stair, symmetryLevel);
        output_stair->SetToReluFunction();
        if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + stair * numStairConvolutions), output_stair->len);
        kernelStartDepth += (startDepth + stair * numStairConvolutions) * numStairConvolutions;
    }

    DeleteOnlyShell(input_stair);
    DeleteOnlyShell(output_stair);
    DeleteOnlyShell(kernel_stair);
    DeleteOnlyShell(bias_stair);
}


void BackwardStairsSymmetricConvolutionRelu(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, int symmetryLevel){

    tensor* input_stair = new tensor();
    tensor* inputDelta_stair = new tensor();
    tensor* output_stair = new tensor();
    tensor* outputDelta_stair = new tensor();
    tensor* kernel_stair = new tensor();
    tensor* kernelGrad_stair = new tensor();
    vect* biasGrad_stair = new vect(0);
    int kernelStartDepth = kernel->depth;

    for(int stair=numStairs - 1; stair>=0; --stair){
        input_stair->SubTensor(input, startDepth + stair * numStairConvolutions);
        inputDelta_stair->SubTensor(inputDelta, startDepth + stair * numStairConvolutions);

        output_stair->SubTensor(input, startDepth + stair * numStairConvolutions, numStairConvolutions);
        outputDelta_stair->SubTensor(inputDelta, startDepth + stair * numStairConvolutions, numStairConvolutions);

        kernelStartDepth -= (startDepth + stair * numStairConvolutions) * numStairConvolutions;
        kernel_stair->SubTensor(kernel, kernelStartDepth, (startDepth + stair * numStairConvolutions) * numStairConvolutions);
        kernelGrad_stair->SubTensor(kernelGrad, kernelStartDepth, (startDepth + stair * numStairConvolutions) * numStairConvolutions);

        outputDelta_stair->BackwardRelu(output_stair);
        BackwardSymmetricConvolution3D(input_stair, inputDelta_stair, outputDelta_stair, kernel_stair, kernelGrad_stair, biasGrad_stair, symmetryLevel);
        inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + stair * numStairConvolutions));
    }
    DeleteOnlyShell(input_stair);
    DeleteOnlyShell(inputDelta_stair);
    DeleteOnlyShell(output_stair);
    DeleteOnlyShell(outputDelta_stair);
    DeleteOnlyShell(kernel_stair);
    DeleteOnlyShell(kernelGrad_stair);
    DeleteOnlyShell(biasGrad_stair);

}







void ForwardStairsFullConvolution(tensor* input, tensor* kernel, vect* bias, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, bool testMode, int symmetryLevel){
    tensor* input_stair = new tensor();
    tensor* output_stair = new tensor();
    tensor* kernel_stair = new tensor();
    tensor* min_output_stair = new tensor();
    vect* bias_stair = new vect(0);
    int kernelStartDepth = 0;

    for(int stair=0; stair<numStairs; ++stair){
        input_stair->SubTensor(input,       startDepth + 2 * stair * numStairConvolutions);
        output_stair->SubTensor(input,      startDepth + 2 * stair * numStairConvolutions,  numStairConvolutions);
        min_output_stair->SubTensor(input,  startDepth + 2 * stair * numStairConvolutions + numStairConvolutions, numStairConvolutions);
        kernel_stair->SubTensor(kernel, kernelStartDepth, (startDepth + 2 * stair * numStairConvolutions) * numStairConvolutions);
        if (bias->len > 0)
            bias_stair->SubVect(bias, stair * numStairConvolutions, numStairConvolutions);
        SymmetricConvolution3D(input_stair, output_stair, kernel_stair, bias_stair, symmetryLevel);
        min_output_stair->BranchMaxMin(output_stair);
        if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * stair * numStairConvolutions), 2 * output_stair->len);
        kernelStartDepth += (startDepth + 2 * stair * numStairConvolutions) * numStairConvolutions;
    }

    DeleteOnlyShell(input_stair);
    DeleteOnlyShell(output_stair);
    DeleteOnlyShell(min_output_stair);
    DeleteOnlyShell(kernel_stair);
    DeleteOnlyShell(bias_stair);
}



void BackwardStairsFullConvolution(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, int symmetryLevel){
    tensor* input_stair = new tensor();
    tensor* inputDelta_stair = new tensor();
    tensor* full_output_stair = new tensor();
    tensor* outputDelta_stair = new tensor();
    tensor* min_outputDelta_stair = new tensor();
    tensor* kernel_stair = new tensor();
    tensor* kernelGrad_stair = new tensor();
    vect* biasGrad_stair = new vect(0);

    int kernelStartDepth = kernel->depth;

    for(int stair=numStairs - 1; stair>=0; --stair){
        input_stair     ->SubTensor(input,      startDepth + 2 * stair * numStairConvolutions);
        inputDelta_stair->SubTensor(inputDelta, startDepth + 2 * stair * numStairConvolutions);

        full_output_stair    ->SubTensor(input,       startDepth + 2 * stair * numStairConvolutions, 2 * numStairConvolutions);
        outputDelta_stair    ->SubTensor(inputDelta,  startDepth + 2 * stair * numStairConvolutions,  numStairConvolutions);
        min_outputDelta_stair->SubTensor(inputDelta,  startDepth + 2 * stair * numStairConvolutions + numStairConvolutions, numStairConvolutions);

        kernelStartDepth -= (startDepth + 2 * stair * numStairConvolutions) * numStairConvolutions;
        kernel_stair    ->SubTensor(kernel,     kernelStartDepth, (startDepth + 2 * stair * numStairConvolutions) * numStairConvolutions);
        kernelGrad_stair->SubTensor(kernelGrad, kernelStartDepth, (startDepth + 2 * stair * numStairConvolutions) * numStairConvolutions);
        if (biasGrad->len > 0)
            biasGrad_stair->SubVect(biasGrad, stair * numStairConvolutions, numStairConvolutions);

        outputDelta_stair->MaxMinBackward(min_outputDelta_stair, full_output_stair);
        BackwardSymmetricConvolution3D(input_stair, inputDelta_stair, outputDelta_stair, kernel_stair, kernelGrad_stair, biasGrad_stair, symmetryLevel);
        inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * stair * numStairConvolutions));
    }
    DeleteOnlyShell(input_stair);
    DeleteOnlyShell(inputDelta_stair);
    DeleteOnlyShell(full_output_stair);
    DeleteOnlyShell(outputDelta_stair);
    DeleteOnlyShell(min_outputDelta_stair);
    DeleteOnlyShell(kernel_stair);
    DeleteOnlyShell(kernelGrad_stair);
    DeleteOnlyShell(biasGrad_stair);
}






void ForwardStairsFullConvolutionRelu(tensor* input, tensor* kernel, vect* bias, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, bool testMode, int symmetryLevel){
    tensor* input_stair = new tensor();
    tensor* output_stair = new tensor();
    tensor* kernel_stair = new tensor();
    vect* bias_stair = new vect(0);
    int kernelStartDepth = 0;

    for(int stair=0; stair<numStairs; ++stair){
        input_stair->SubTensor(input,       startDepth + stair * numStairConvolutions);
        output_stair->SubTensor(input,      startDepth + stair * numStairConvolutions,  numStairConvolutions);
        kernel_stair->SubTensor(kernel, kernelStartDepth, (startDepth + stair * numStairConvolutions) * numStairConvolutions);
        if (bias->len > 0)
            bias_stair->SubVect(bias, stair * numStairConvolutions, numStairConvolutions);
        SymmetricConvolution3D(input_stair, output_stair, kernel_stair, bias_stair, symmetryLevel);
        output_stair->SetToReluFunction();
        if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + stair * numStairConvolutions), output_stair->len);
        kernelStartDepth += (startDepth + stair * numStairConvolutions) * numStairConvolutions;
    }

    DeleteOnlyShell(input_stair);
    DeleteOnlyShell(output_stair);
    DeleteOnlyShell(kernel_stair);
    DeleteOnlyShell(bias_stair);
}



void BackwardStairsFullConvolutionRelu(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, int symmetryLevel){
    tensor* input_stair = new tensor();
    tensor* inputDelta_stair = new tensor();
    tensor* output_stair = new tensor();
    tensor* outputDelta_stair = new tensor();
    tensor* kernel_stair = new tensor();
    tensor* kernelGrad_stair = new tensor();
    vect* biasGrad_stair = new vect(0);

    int kernelStartDepth = kernel->depth;

    for(int stair=numStairs - 1; stair>=0; --stair){
        input_stair     ->SubTensor(input,      startDepth + stair * numStairConvolutions);
        inputDelta_stair->SubTensor(inputDelta, startDepth + stair * numStairConvolutions);

        output_stair     ->SubTensor(input,       startDepth + stair * numStairConvolutions, numStairConvolutions);
        outputDelta_stair->SubTensor(inputDelta,  startDepth + stair * numStairConvolutions,  numStairConvolutions);

        kernelStartDepth -= (startDepth + stair * numStairConvolutions) * numStairConvolutions;
        kernel_stair    ->SubTensor(kernel,     kernelStartDepth, (startDepth + stair * numStairConvolutions) * numStairConvolutions);
        kernelGrad_stair->SubTensor(kernelGrad, kernelStartDepth, (startDepth + stair * numStairConvolutions) * numStairConvolutions);
        if (biasGrad->len > 0)
            biasGrad_stair->SubVect(biasGrad, stair * numStairConvolutions, numStairConvolutions);

        outputDelta_stair->BackwardRelu(output_stair);
        BackwardSymmetricConvolution3D(input_stair, inputDelta_stair, outputDelta_stair, kernel_stair, kernelGrad_stair, biasGrad_stair, symmetryLevel);
        inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + stair * numStairConvolutions));
    }
    DeleteOnlyShell(input_stair);
    DeleteOnlyShell(inputDelta_stair);
    DeleteOnlyShell(output_stair);
    DeleteOnlyShell(outputDelta_stair);
    DeleteOnlyShell(kernel_stair);
    DeleteOnlyShell(kernelGrad_stair);
    DeleteOnlyShell(biasGrad_stair);
}







void ForwardStairsSequentialConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, int* indices, bool testMode, int symmetryLevel){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    int vertKernelStartIndex = 0;
    int * indices_j;
    int j=0;

    for(int conv=0; conv<numStairConvolutions; ++conv){
        j = conv;
        indices_j = indices + vertKernelStartIndex;
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, startDepth);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        SymmetricConvolution(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, symmetryLevel);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
        vertKernelStartIndex += startDepth;
    }

    for(int stair=1; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            j = stair * numStairConvolutions + conv;
            indices_j = indices + vertKernelStartIndex;
            vertOutput_j->SetToTensorLayer(verticalConv, j);
            convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
            minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
            kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, 2 * numStairConvolutions);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            SymmetricConvolution(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, symmetryLevel);
            minOutput_j->BranchMaxMin(convOutput_j);
            if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
            vertKernelStartIndex += 2 * numStairConvolutions;
        }

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
}






void ForwardStairsPyramidalConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, int* indices, bool testMode){

    verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    int vertKernelStartIndex = 0;
    int * indices_j;
    int j=0;
    for(int stair=0; stair<numStairs; ++stair)
        for(int conv=0; conv<numStairConvolutions; ++conv){
            j = stair * numStairConvolutions + conv;
            indices_j = indices + vertKernelStartIndex;
            vertOutput_j->SetToTensorLayer(verticalConv, j);
            convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
            minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
            kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, startDepth + 2 * numStairConvolutions * stair);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);

            PyramidalConvolution(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, stair);
            //BottleneckConvolutionStandardRandomSymmetricNoBiasDrop(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, verticalConvActivity_j, testMode);
            minOutput_j->BranchMaxMin(convOutput_j);
            if (!testMode)
                input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
            vertKernelStartIndex += startDepth + 2 * numStairConvolutions * stair;
        }

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
}








void SequentialBottleneckConvolutionStandardRandomFullySymmetric(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices, bool testMode){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();
    realNumber* biasHor_j = new realNumber();

    int vertKernelStartIndex = 0;
    int * indices_j;

    for(int j=0; j<nConvolutions; ++j){
        indices_j = indices + vertKernelStartIndex;
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        vertKernelStartIndex += alwaysPresentDepth + min(startDepth - alwaysPresentDepth + 2 * j, limitDepth);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        biasHor_j->SetToVectElement(bias_hor, j);

        BottleneckConvolutionStandardRandomFullySymmetric(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, biasHor_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
    }
    if (vertKernelStartIndex != kernel_vert->len)
        cout<<"Error in forward random pass"<<endl;

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(biasHor_j);
}



void SequentialBottleneckConvolutionStandardRandomLimited(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices, bool testMode){

    //verticalConv->SetToZero();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    matrix* vertOutput_j = new matrix();
    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();
    realNumber* biasHor_j = new realNumber();

    int vertKernelStartIndex = 0;
    int * indices_j;

    for(int j=0; j<nConvolutions; ++j){
        indices_j = indices + vertKernelStartIndex;
        vertOutput_j->SetToTensorLayer(verticalConv, j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        kernelVert_j->SubTensor(kernel_vert, vertKernelStartIndex, alwaysPresentDepth + limitDepth);
        vertKernelStartIndex += alwaysPresentDepth + limitDepth;
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        biasHor_j->SetToVectElement(bias_hor, j);

        BottleneckConvolutionStandardRandom(input, indices_j, vertOutput_j, convOutput_j, kernelVert_j, kernelHor_j, biasHor_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        if (!testMode)
            input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * convOutput_j->len);
    }
    if (vertKernelStartIndex != kernel_vert->len)
        cout<<"Error in forward random limited pass"<<endl;

    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(biasHor_j);
}





void SequentialConvolutionMultipleStandard(tensor* input, tensor* kernel, vect* bias, int startDepth, int nConvolutions, activityData* inputActivity){
    tensor* input_j = new tensor();
    tensor* convOutput_j = new tensor();
    tensor* minOutput_j = new tensor();
    tensor* kernel_j = new tensor();
    vect* bias_j = new vect();

    for(int j=0; j<nConvolutions; ++j){
        input_j->SubTensor(input, startDepth *power(3, j) );
        convOutput_j->SubTensor(input, startDepth *power(3, j), startDepth *power(3, j));
        minOutput_j->SubTensor(input, 2 * startDepth *power(3, j), startDepth *power(3, j));
        kernel_j->SubTensor(kernel, startDepth * (power(3, j) - 1) / 2, startDepth *power(3, j));
        bias_j->SubVect(bias, startDepth * (power(3, j) - 1) / 2, startDepth *power(3, j));

        ConvoluteMultipleStandard3D3D(input_j, convOutput_j, kernel_j, bias_j);
        minOutput_j->BranchMaxMin(convOutput_j);
        //minOutput_j->SetToMinReluFunction(convOutput_j);
        //convOutput_j->SetToReluFunction();
        input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth *power(3, j)), 2 * convOutput_j->len);
    }

    DeleteOnlyShell(input_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(kernel_j);
    DeleteOnlyShell(bias_j);
}






void SequentialConvolution(tensor* input, tensor* reversedKernel, vect* bias, int paddingR, int paddingC, vector<int> &indexInputRow,
                           int startDepth, activityData* inputActivity){
    int convolutionNumber = bias->len;

    tensor* input_j = new tensor();
    matrix* convOutput_j = new matrix();
    matrix* minOutput_j = new matrix();
    tensor* rKernel_j = new tensor();
    realNumber* bias_j = new realNumber();

    for(int j=0; j<convolutionNumber; ++j){
        input_j->SubTensor(input, startDepth + 2 * j);
        convOutput_j->SetToTensorLayer(input, startDepth + 2 * j);
        minOutput_j->SetToTensorLayer(input, startDepth + 2 * j + 1);
        rKernel_j->SubTensor(reversedKernel, j * (startDepth + j - 1), startDepth + 2 * j);
        bias_j->SetToVectElement(bias, j);

        Convolute3D2D(input_j, convOutput_j, rKernel_j, bias_j, paddingR, paddingC, indexInputRow);
        minOutput_j->SetToMinReluFunction(convOutput_j);
        convOutput_j->SetToReluFunction();
        input->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * j), 2 * input->rows * input->cols);
    }

    DeleteOnlyShell(input_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(minOutput_j);
    DeleteOnlyShell(rKernel_j);
    DeleteOnlyShell(bias_j);
}


void BackwardSequentialConvolutionStandard(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                                           int startDepth, bool computeLastDelta, activityData* inputActivity){
    int convolutionNumber = biasGrad->len;

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    tensor* kernel_j = new tensor();
    tensor* kernelGrad_j = new tensor();
    realNumber* biasGrad_j = new realNumber();

    for(int j = convolutionNumber - 1; j>=0; --j){
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);
        input_j->SubTensor(input, startDepth + 2 * j);
        inputDelta_j->SubTensor(inputDelta, startDepth + 2 * j);
        kernel_j->SubTensor(kernel, j * (startDepth + j - 1), startDepth + 2 * j);
        kernelGrad_j->SubTensor(kernelGrad, j * (startDepth + j - 1), startDepth + 2 * j);
        biasGrad_j->SetToVectElement(biasGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        if (j==0 && !computeLastDelta)
            BackwardConvoluteGradStandard3D2D(input_j, convOutputDelta_j, kernelGrad_j, biasGrad_j);
        else
            BackwardConvoluteStandard3D2D(input_j, inputDelta_j, convOutputDelta_j, kernel_j, kernelGrad_j, biasGrad_j);

        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_j);
    DeleteOnlyShell(kernelGrad_j);
    DeleteOnlyShell(biasGrad_j);
}


void BackwardSequentialMaxMinStandard(tensor* input, tensor* inputDelta, vect* kernel, vect* kernelGrad, vect* biasGrad, int startDepth, int nConvolutions, activityData* inputActivity){
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    vect* kernel_j = new vect();
    vect* kernelGrad_j = new vect();

    for(int j = nConvolutions - 1; j>=0; --j){
        inputDelta->elem[startDepth + 2 * j + 1]    *= (input->elem[startDepth + 2 * j + 1] < 0);
        inputDelta->elem[startDepth + 2 * j]        *= (input->elem[startDepth + 2 * j] > 0);
        inputDelta->elem[startDepth + 2 * j] += inputDelta->elem[startDepth + 2 * j + 1];

        input_j->SubTensor(input, startDepth + 2 * j);
        inputDelta_j->SubTensor(inputDelta, startDepth + 2 * j);
        kernel_j->SubVect(kernel, j * (startDepth + j - 1), startDepth + 2 * j);
        kernelGrad_j->SubVect(kernelGrad, j * (startDepth + j - 1), startDepth + 2 * j);

        kernelGrad_j->orderedData::Add(inputDelta->elem[startDepth + 2 * j], input_j);
        biasGrad->elem[j] += inputDelta->elem[startDepth + 2 * j];
        inputDelta_j->Add(inputDelta->elem[startDepth + 2 * j], kernel_j);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth + 2 * (j-1), 2);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth);
    }

    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_j);
    DeleteOnlyShell(kernelGrad_j);
}




void BackwardSequentialBottleneckConvolutionReluStandard(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, activityData* inputActivity){
    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* convOutput_j = new matrix();
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    tensor* kernel_vert_j = new tensor();
    matrix* kernel_hor_j = new matrix();
    tensor* kernelGrad_vert_j = new tensor();
    matrix* kernelGrad_hor_j = new matrix();
    realNumber* biasGrad_hor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    for(int j = nConvolutions - 1; j>=0; --j){
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + j);
        convOutput_j->SetToTensorLayer(input, startDepth + j);
        input_j->SubTensor(input, startDepth + j);
        inputDelta_j->SubTensor(inputDelta, startDepth + j);
        kernel_vert_j->SubTensor(kernel_vert, j * (2 * startDepth + j - 1) / 2, startDepth + j);
        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
        kernelGrad_vert_j->SubTensor(kernelGrad_vert, j * (2 * startDepth + j - 1) / 2, startDepth + j);
        kernelGrad_hor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGrad_hor_j->SetToVectElement(biasGrad_hor, j);
        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->BackwardRelu(convOutput_j);
        BackwardBottleneckConvolutionStandard(input_j, inputDelta_j, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                            kernel_vert_j, kernel_hor_j, kernelGrad_vert_j, kernelGrad_hor_j, biasGrad_hor_j);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + (j-1) ), convOutput_j->len);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(convOutput_j);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_vert_j);
    DeleteOnlyShell(kernel_hor_j);
    DeleteOnlyShell(kernelGrad_vert_j);
    DeleteOnlyShell(kernelGrad_hor_j);
    DeleteOnlyShell(biasGrad_hor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);

}




void BackwardSequentialBottleneckConvolutionStandard(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, activityData* inputActivity){
    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    tensor* kernel_vert_j = new tensor();
    matrix* kernel_hor_j = new matrix();
    tensor* kernelGrad_vert_j = new tensor();
    matrix* kernelGrad_hor_j = new matrix();
    realNumber* biasGrad_hor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    for(int j = nConvolutions - 1; j>=0; --j){
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);
        input_j->SubTensor(input, startDepth + 2 * j);
        inputDelta_j->SubTensor(inputDelta, startDepth + 2 * j);
        kernel_vert_j->SubTensor(kernel_vert, j * (startDepth + j - 1), startDepth + 2 * j);
        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
        kernelGrad_vert_j->SubTensor(kernelGrad_vert, j * (startDepth + j - 1), startDepth + 2 * j);
        kernelGrad_hor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGrad_hor_j->SetToVectElement(biasGrad_hor, j);
        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        //if (!computeDelta && j==0)
        //    BackwardBottleneckConvolutionStandardGrad(input_j, inputDelta_j, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
        //                    kernel_vert_j, kernel_hor_j, kernelGrad_vert_j, kernelGrad_hor_j, biasGrad_hor_j);
        //else
        BackwardBottleneckConvolutionStandard(input_j, inputDelta_j, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                            kernel_vert_j, kernel_hor_j, kernelGrad_vert_j, kernelGrad_hor_j, biasGrad_hor_j);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_vert_j);
    DeleteOnlyShell(kernel_hor_j);
    DeleteOnlyShell(kernelGrad_vert_j);
    DeleteOnlyShell(kernelGrad_hor_j);
    DeleteOnlyShell(biasGrad_hor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}







void BackwardSequentialBottleneckConvolutionStandardLimited(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity){
    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();
    tensor* inputStart = new tensor();
    tensor* inputStartDelta = new tensor();
    tensor* inputLast_j = new tensor();
    tensor* inputLastDelta_j = new tensor();

    tensor* kernelVertStart_j = new tensor();
    tensor* kernelVertLast_j = new tensor();

    matrix* kernel_hor_j = new matrix();

    tensor* kernelGradVertStart_j = new tensor();
    tensor* kernelGradVertLast_j = new tensor();

    matrix* kernelGradHor_j = new matrix();
    realNumber* biasGradHor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    inputStart->SubTensor(input, alwaysPresentDepth);
    inputStartDelta->SubTensor(inputDelta, alwaysPresentDepth);

    int vertKernelEndIndex = kernel_vert->len;

    for(int j = nConvolutions - 1; j>=0; --j){
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);
        inputLast_j->SubTensor(input, startDepth + 2 * j - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth), min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));
        inputLastDelta_j->SubTensor(inputDelta, startDepth + 2 * j - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth), min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));

        kernelVertStart_j->SubTensor(kernel_vert, vertKernelEndIndex - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) - alwaysPresentDepth, alwaysPresentDepth);
        kernelVertLast_j->SubTensor(kernel_vert, vertKernelEndIndex - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth), min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));

        kernel_hor_j->SetToTensorLayer(kernel_hor, j);

        kernelGradVertStart_j->SubTensor(kernelGrad_vert, vertKernelEndIndex - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) - alwaysPresentDepth, alwaysPresentDepth);
        kernelGradVertLast_j->SubTensor(kernelGrad_vert, vertKernelEndIndex - min(startDepth - alwaysPresentDepth + 2 * j, limitDepth), min(startDepth - alwaysPresentDepth + 2 * j, limitDepth));

        vertKernelEndIndex -= min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) + alwaysPresentDepth;

        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

        biasGradHor_j->SetToVectElement(biasGrad_hor, j);
        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

        BackwardBottleneckConvolutionStandardLimitedFull(inputStart, inputLast_j, inputStartDelta, inputLastDelta_j, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                            kernelVertStart_j, kernelVertLast_j, kernel_hor_j, kernelGradVertStart_j, kernelGradVertLast_j, kernelGradHor_j, biasGradHor_j);

        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(inputStart);
    DeleteOnlyShell(inputLast_j);
    DeleteOnlyShell(inputStartDelta);
    DeleteOnlyShell(inputLastDelta_j);
    DeleteOnlyShell(kernelVertStart_j);
    DeleteOnlyShell(kernelVertLast_j);
    DeleteOnlyShell(kernel_hor_j);
    DeleteOnlyShell(kernelGradVertStart_j);
    DeleteOnlyShell(kernelGradVertLast_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(biasGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}








void BackwardSequentialBottleneckConvolutionStandardRandom(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices){

    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    realNumber* biasGradHor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;

    for(int j = nConvolutions - 1; j>=0; --j){
        dependencyNum = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) + alwaysPresentDepth;
        vertKernelEndIndex -= dependencyNum;

        indices_j = indices + vertKernelEndIndex;
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);

        kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGradHor_j->SetToVectElement(biasGrad_hor, j);

        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

        BackwardBottleneckConvolutionStandardRandom(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                                    kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, biasGradHor_j);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }
    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(biasGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}




void BackwardSequentialBottleneckConvolutionStandardRandomSymmetric(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution, tensor* vertConvolutionGrad, int startDepth,
            int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, activityData* verticalConvActivity, int* indices){

    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    realNumber* biasGradHor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();
    activityData * verticalConvActivity_j = new activityData();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;

    for(int j = nConvolutions - 1; j>=0; --j){
        dependencyNum = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) + alwaysPresentDepth;
        vertKernelEndIndex -= dependencyNum;

        indices_j = indices + vertKernelEndIndex;
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);

        kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGradHor_j->SetToVectElement(biasGrad_hor, j);

        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        verticalConvActivity_j->SubActivityData(verticalConvActivity, j * vertOutput_j->len, vertOutput_j->len);

        BackwardBottleneckConvolutionStandardRandomSymmetricDrop(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                                    kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, biasGradHor_j, verticalConvActivity_j);

        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }
    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(biasGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
    DeleteOnlyShellActivity(verticalConvActivity_j);
}





void BackwardSequentialBottleneckConvolutionStandardRandomSymmetricNoBias(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
            tensor* kernel_hor, tensor* kernelGrad_hor, tensor* vertConvolution, tensor* vertConvolutionGrad, int startDepth,
            int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, activityData* verticalConvActivity, int* indices){

    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();
    activityData * verticalConvActivity_j = new activityData();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;

    for(int j = nConvolutions - 1; j>=0; --j){
        dependencyNum = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) + alwaysPresentDepth;
        vertKernelEndIndex -= dependencyNum;

        indices_j = indices + vertKernelEndIndex;
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);

        kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        verticalConvActivity_j->SubActivityData(verticalConvActivity, j * vertOutput_j->len, vertOutput_j->len);

        BackwardBottleneckConvolutionStandardRandomSymmetricNoBiasDrop(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                                    kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, verticalConvActivity_j);

        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }
    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
    DeleteOnlyShellActivity(verticalConvActivity_j);
}






void BackwardStairsConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, activityData* verticalConvActivity, int* indices){

    verticalConvDelta->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();
    activityData * verticalConvActivity_j = new activityData();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;
    int j = 0;

    for(int stair=numStairs-1; stair>=0; --stair)
        for(int conv=numStairConvolutions-1; conv>=0; --conv){
            j = stair * numStairConvolutions + conv;
            dependencyNum = startDepth + 2 * numStairConvolutions * stair;
            vertKernelEndIndex -= dependencyNum;

            indices_j = indices + vertKernelEndIndex;
            convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
            minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
            output_j->SubTensor(input, startDepth + 2 * j, 2);

            kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
            kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

            vertOutput_j->SetToTensorLayer(verticalConv, j);
            vertOutputDelta_j->SetToTensorLayer(verticalConvDelta, j);

            convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
            verticalConvActivity_j->SubActivityData(verticalConvActivity, j * vertOutput_j->len, vertOutput_j->len);

            BackwardBottleneckConvolutionStandardRandomSymmetricNoBiasDrop(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, verticalConvActivity_j);

            if (j!=0)
                inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
            else
                inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
        }

    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
    DeleteOnlyShellActivity(verticalConvActivity_j);
}






void BackwardStairsSymmetricConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, int* indices, int symmetryLevel){

    verticalConvDelta->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;
    int j = 0;

    for(int stair=numStairs-1; stair>=0; --stair){
        for(int conv=numStairConvolutions-1; conv>=0; --conv){
            j = stair * numStairConvolutions + conv;
            dependencyNum = startDepth + 2 * numStairConvolutions * stair;
            vertKernelEndIndex -= dependencyNum;

            indices_j = indices + vertKernelEndIndex;
            convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
            minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
            output_j->SubTensor(input, startDepth + 2 * j, 2);

            kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
            kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

            vertOutput_j->SetToTensorLayer(verticalConv, j);
            vertOutputDelta_j->SetToTensorLayer(verticalConvDelta, j);

            convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

            BackwardSymmetricConvolution(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, symmetryLevel);
        }
        inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startDepth + 2 * numStairConvolutions * stair));
    }

    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}






void BackwardStairsSequentialConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, int* indices, int symmetryLevel){

    verticalConvDelta->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;
    int j = 0;



    for(int stair=numStairs-1; stair>=1; --stair){
        for(int conv=numStairConvolutions-1; conv>=0; --conv){
            j = stair * numStairConvolutions + conv;
            dependencyNum = 2 * numStairConvolutions;
            vertKernelEndIndex -= dependencyNum;

            indices_j = indices + vertKernelEndIndex;
            convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
            minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
            output_j->SubTensor(input, startDepth + 2 * j, 2);

            kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
            kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

            vertOutput_j->SetToTensorLayer(verticalConv, j);
            vertOutputDelta_j->SetToTensorLayer(verticalConvDelta, j);

            convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

            BackwardSymmetricConvolution(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, symmetryLevel);

            if (j!=0)
                inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
            else
                inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
        }
        inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startDepth + 2 * numStairConvolutions * stair));
    }




    for(int conv=numStairConvolutions-1; conv>=0; --conv){
        j = conv;
        dependencyNum = startDepth;
        vertKernelEndIndex -= dependencyNum;

        indices_j = indices + vertKernelEndIndex;
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);

        kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

        vertOutput_j->SetToTensorLayer(verticalConv, j);
        vertOutputDelta_j->SetToTensorLayer(verticalConvDelta, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

        BackwardSymmetricConvolution(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                            kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, symmetryLevel);

        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }
    inputDelta->SetDroppedElementsToZero(inputActivity, inputDelta->Ind(startDepth));

    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}








void BackwardStairsPyramidalConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, int* indices){

    verticalConvDelta->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;
    int j = 0;

    for(int stair=numStairs-1; stair>=0; --stair)
        for(int conv=numStairConvolutions-1; conv>=0; --conv){
            j = stair * numStairConvolutions + conv;
            dependencyNum = startDepth + 2 * numStairConvolutions * stair;
            vertKernelEndIndex -= dependencyNum;

            indices_j = indices + vertKernelEndIndex;
            convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
            minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
            output_j->SubTensor(input, startDepth + 2 * j, 2);

            kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
            kernelHor_j->SetToTensorLayer(kernel_hor, j);
            kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
            kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);

            vertOutput_j->SetToTensorLayer(verticalConv, j);
            vertOutputDelta_j->SetToTensorLayer(verticalConvDelta, j);

            convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

            BackwardPyramidalConvolution(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, stair);

            if (j!=0)
                inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
            else
                inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
        }

    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}









void BackwardSequentialBottleneckConvolutionStandardRandomFullySymmetric(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices){

    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    realNumber* biasGradHor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;

    for(int j = nConvolutions - 1; j>=0; --j){
        dependencyNum = min(startDepth - alwaysPresentDepth + 2 * j, limitDepth) + alwaysPresentDepth;
        vertKernelEndIndex -= dependencyNum;

        indices_j = indices + vertKernelEndIndex;
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);

        kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGradHor_j->SetToVectElement(biasGrad_hor, j);

        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

        BackwardBottleneckConvolutionStandardRandomFullySymmetric(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                                    kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, biasGradHor_j);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }
    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(biasGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}







void BackwardSequentialBottleneckConvolutionStandardRandomLimited(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices){

    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();

    tensor* kernelVert_j = new tensor();
    matrix* kernelHor_j = new matrix();

    tensor* kernelGradVert_j = new tensor();
    matrix* kernelGradHor_j = new matrix();
    realNumber* biasGradHor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    int vertKernelEndIndex = kernel_vert->len;

    int * indices_j;
    int dependencyNum;

    for(int j = nConvolutions - 1; j>=0; --j){
        dependencyNum = limitDepth + alwaysPresentDepth;
        vertKernelEndIndex -= dependencyNum;

        indices_j = indices + vertKernelEndIndex;
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);

        kernelVert_j->SubTensor(kernel_vert, vertKernelEndIndex, dependencyNum);
        kernelHor_j->SetToTensorLayer(kernel_hor, j);
        kernelGradVert_j->SubTensor(kernelGrad_vert, vertKernelEndIndex, dependencyNum);
        kernelGradHor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGradHor_j->SetToVectElement(biasGrad_hor, j);

        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);

        BackwardBottleneckConvolutionStandardRandom(input, indices_j, inputDelta, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                                                    kernelVert_j, kernelHor_j, kernelGradVert_j, kernelGradHor_j, biasGradHor_j);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }

    if (vertKernelEndIndex != 0)
        cout<<"ERROR in BACKWARD Random Limited"<<endl;
    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(kernelVert_j);
    DeleteOnlyShell(kernelHor_j);
    DeleteOnlyShell(kernelGradVert_j);
    DeleteOnlyShell(kernelGradHor_j);
    DeleteOnlyShell(biasGradHor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);
}









void BackwardSequentialBottleneckConvolutionStandardPartialGrad(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, activityData* inputActivity){

    vertConvolutionGrad->SetToZero();

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    tensor* kernel_vert_j = new tensor();
    matrix* kernel_hor_j = new matrix();
    tensor* kernelGrad_vert_j = new tensor();
    matrix* kernelGrad_hor_j = new matrix();
    realNumber* biasGrad_hor_j = new realNumber();
    matrix* vertOutput_j = new matrix();
    matrix* vertOutputDelta_j = new matrix();

    for(int j = nConvolutions - 1; j>=0; --j){
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);
        input_j->SubTensor(input, startDepth + 2 * j);
        inputDelta_j->SubTensor(inputDelta, startDepth + 2 * j);
        kernel_vert_j->SubTensor(kernel_vert, j * (startDepth + j - 1), startDepth + 2 * j);
        kernel_hor_j->SetToTensorLayer(kernel_hor, j);
        kernelGrad_vert_j->SubTensor(kernelGrad_vert, j * (startDepth + j - 1), startDepth + 2 * j);
        kernelGrad_hor_j->SetToTensorLayer(kernelGrad_hor, j);
        biasGrad_hor_j->SetToVectElement(biasGrad_hor, j);
        vertOutput_j->SetToTensorLayer(vertConvolution, j);
        vertOutputDelta_j->SetToTensorLayer(vertConvolutionGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        //if (!computeDelta && j==0)
        //    BackwardBottleneckConvolutionStandardGrad(input_j, inputDelta_j, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
        //                    kernel_vert_j, kernel_hor_j, kernelGrad_vert_j, kernelGrad_hor_j, biasGrad_hor_j);
        //else
        BackwardBottleneckConvolutionStandardPartialGrad(input_j, inputDelta_j, convOutputDelta_j, vertOutput_j, vertOutputDelta_j,
                            kernel_vert_j, kernel_hor_j, kernelGrad_vert_j, kernelGrad_hor_j, biasGrad_hor_j, startDepth);
        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_vert_j);
    DeleteOnlyShell(kernel_hor_j);
    DeleteOnlyShell(kernelGrad_vert_j);
    DeleteOnlyShell(kernelGrad_hor_j);
    DeleteOnlyShell(biasGrad_hor_j);
    DeleteOnlyShell(vertOutput_j);
    DeleteOnlyShell(vertOutputDelta_j);

}










void BackwardSequentialConvolutionMultipleStandard(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                                   int startDepth, int nConvolutions, bool computeLastDelta, activityData* inputActivity){
    tensor* convOutputDelta_j = new tensor();
    tensor* minOutputDelta_j = new tensor();
    tensor* output_j = new tensor();
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    tensor* kernel_j = new tensor();
    tensor* kernelGrad_j = new tensor();
    vect* biasGrad_j = new vect();

    for(int j = nConvolutions - 1; j>=0; --j){
        convOutputDelta_j->SubTensor(inputDelta, startDepth *power(3, j), startDepth *power(3, j));
        minOutputDelta_j->SubTensor(inputDelta, 2 * startDepth *power(3, j), startDepth *power(3, j));
        output_j->SubTensor(input, startDepth *power(3, j), 2 * startDepth *power(3, j));
        input_j->SubTensor(input, startDepth *power(3, j));
        inputDelta_j->SubTensor(inputDelta, startDepth *power(3, j));
        kernel_j->SubTensor(kernel, startDepth * (power(3, j) - 1) / 2, startDepth *power(3, j));
        kernelGrad_j->SubTensor(kernelGrad, startDepth * (power(3, j) - 1) / 2, startDepth *power(3, j));
        biasGrad_j->SubVect(biasGrad, startDepth * (power(3, j) - 1) / 2, startDepth *power(3, j));

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        BackwardConvoluteMultipleStandard3D3D(input_j, inputDelta_j, convOutputDelta_j,
                                              kernel_j, kernelGrad_j, biasGrad_j);

        inputDelta->SetDroppedElementsToZero(inputActivity);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_j);
    DeleteOnlyShell(kernelGrad_j);
    DeleteOnlyShell(biasGrad_j);
}







void BackwardSequentialConvolution(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                                   int paddingR, int paddingC, vector<int> & indexInputRow, int startDepth, bool computeLastDelta, activityData* inputActivity){
    int convolutionNumber = biasGrad->len;

    matrix* convOutputDelta_j = new matrix();
    matrix* minOutputDelta_j = new matrix();
    tensor* output_j = new tensor();
    tensor* input_j = new tensor();
    tensor* inputDelta_j = new tensor();
    tensor* kernel_j = new tensor();
    tensor* kernelGrad_j = new tensor();
    realNumber* biasGrad_j = new realNumber();

    for(int j = convolutionNumber - 1; j>=0; --j){
        convOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j);
        minOutputDelta_j->SetToTensorLayer(inputDelta, startDepth + 2 * j + 1);
        output_j->SubTensor(input, startDepth + 2 * j, 2);
        input_j->SubTensor(input, startDepth + 2 * j);
        inputDelta_j->SubTensor(inputDelta, startDepth + 2 * j);
        kernel_j->SubTensor(kernel, j * (startDepth + j - 1), startDepth + 2 * j);
        kernelGrad_j->SubTensor(kernelGrad, j * (startDepth + j - 1), startDepth + 2 * j);
        biasGrad_j->SetToVectElement(biasGrad, j);

        convOutputDelta_j->MaxMinBackward(minOutputDelta_j, output_j);
        if (j==0 && !computeLastDelta)
            BackwardConvoluteGrad3D2D(input_j, convOutputDelta_j, kernelGrad_j, biasGrad_j, paddingR, paddingC, indexInputRow);
        else
            BackwardConvolute3D2D(input_j, inputDelta_j, convOutputDelta_j, kernel_j, kernelGrad_j, biasGrad_j, paddingR, paddingC, indexInputRow);

        if (j!=0)
            inputDelta->SetDroppedElementsToZero(inputActivity, input->Ind(startDepth + 2 * (j-1) ), 2 * input->rows * input->cols);
        else
            inputDelta->SetDroppedElementsToZero(inputActivity, startDepth * input->rows * input->cols);
    }

    DeleteOnlyShell(convOutputDelta_j);
    DeleteOnlyShell(minOutputDelta_j);
    DeleteOnlyShell(output_j);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(kernel_j);
    DeleteOnlyShell(kernelGrad_j);
    DeleteOnlyShell(biasGrad_j);
}




void AveragePool2D(matrix* input, matrix* output, int kernelR, int kernelC){
    double fact = 1.0/(kernelR*kernelC);
    int minIR, maxIR;
    int minIC, maxIC;
    double * out_oR, * in_iR;
    //double temp;
    //bool * outAct_oR;
    //output->SetToZero();
    for(int oR=0; oR<output->rows; ++oR){
        minIR = oR * kernelR;
        maxIR = minIR + kernelR - 1;
        out_oR = output->Row(oR);
        //outAct_oR = outputActivity->Row(oR);

        for(int iR=minIR; iR<=maxIR; ++iR){

            in_iR = input->Row(iR);

            for(int oC=0; oC<output->cols; ++oC){
                //if (!outAct_oR[oC]) continue;
                minIC = oC*kernelC;
                maxIC = minIC + kernelC - 1;
                for(int iC=minIC; iC<=maxIC; ++iC)
                    out_oR[oC] += in_iR[iC];
            }
        }
    }

    output->Multiply(fact);
}


void AveragePool2D_2_2(matrix* input, matrix* output){
    double fact = 0.25;
    double * out_oR, * in_iR;
    int out_rows = output->rows;
    int out_cols = output->cols;

    for(int oR=0; oR<out_rows; ++oR){
        out_oR = output->Row(oR);
        in_iR = input->Row(oR * 2);
        for(int oC=0; oC<out_cols; ++oC){
            out_oR[oC] += in_iR[oC * 2] + in_iR[oC * 2 + 1];
        }

        in_iR = input->Row(oR * 2 + 1);
        for(int oC=0; oC<out_cols; ++oC){
            out_oR[oC] += in_iR[oC * 2] + in_iR[oC * 2 + 1];
        }
    }
    output->Multiply(fact);
}


void BackwardAveragePool2D(matrix* inputDelta, matrix* outputDelta, int kernelR, int kernelC){
    double fact = 1.0/(kernelR*kernelC);

    int minIR, maxIR;
    int minIC, maxIC;
    double * outDelta_oR, * inDelta_iR;
    //double temp;
    //bool * outAct_oR;

    for(int oR=0; oR<outputDelta->rows; ++oR){
        minIR = oR * kernelR;
        maxIR = minIR + kernelR - 1;
        outDelta_oR = outputDelta->Row(oR);
        //outAct_oR = outputActivity->Row(oR);

        for(int iR=minIR; iR<=maxIR; ++iR){

            inDelta_iR = inputDelta->Row(iR);

            for(int oC=0; oC<outputDelta->cols; ++oC){
                //if (!outAct_oR[oC]) continue;
                minIC = oC*kernelC;
                maxIC = minIC + kernelC - 1;
                for(int iC=minIC; iC<=maxIC; ++iC)
                    inDelta_iR[iC] += outDelta_oR[oC];
            }
        }
    }
    inputDelta->Multiply(fact);
}

void BackwardAveragePool2D_2_2(matrix* inputDelta, matrix* outputDelta){
    double fact = 0.25;
    double * outDelta_oR, * inDelta_iR;
    int out_rows = outputDelta->rows;
    int out_cols = outputDelta->cols;

    for(int oR=0; oR<out_rows; ++oR){
        outDelta_oR = outputDelta->Row(oR);

        inDelta_iR = inputDelta->Row(oR * 2);
        for(int oC=0; oC<out_cols; ++oC){
            inDelta_iR[oC * 2]      += outDelta_oR[oC] * fact;
            inDelta_iR[oC * 2 + 1]  += outDelta_oR[oC] * fact;
        }

        inDelta_iR = inputDelta->Row(oR * 2 + 1);
        for(int oC=0; oC<out_cols; ++oC){
            inDelta_iR[oC * 2]      += outDelta_oR[oC] * fact;
            inDelta_iR[oC * 2 + 1]  += outDelta_oR[oC] * fact;
        }
    }
}


void AveragePool3D_2_2(tensor* input, tensor* output){
    matrix* input_d = new matrix();
    matrix* output_d = new matrix();

    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output_d->SetToTensorLayer(output, d);
        AveragePool2D_2_2(input_d, output_d);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(output_d);
}

void AveragePool3D_all(tensor* input, tensor* output){
    double fact = 1.0/(input->rows * input->cols);
    matrix* input_d = new matrix();

    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output->elem[d] = input_d->Sum();
    }

    output->Multiply(fact);
    DeleteOnlyShell(input_d);
}

void AveragePool3D_all(tensor* input, tensor* output, int numNonZero){
    double fact = 1.0/(numNonZero);
    matrix* input_d = new matrix();

    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output->elem[d] = input_d->Sum();
    }

    output->Multiply(fact);
    DeleteOnlyShell(input_d);
}

void AveragePool3D(tensor* input, tensor* output, int kernelR, int kernelC){
    if (kernelR == 2 && kernelC == 2){
        AveragePool3D_2_2(input, output);
        return;
    }

    if (input->rows == kernelR && input->cols == kernelC){
        AveragePool3D_all(input, output);
        return;
    }
    matrix* input_d = new matrix();
    matrix* output_d = new matrix();

    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output_d->SetToTensorLayer(output, d);

        AveragePool2D(input_d, output_d, kernelR, kernelC);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(output_d);
}


void BackwardAveragePool3D_2_2(tensor* inputDelta, tensor* outputDelta){
    matrix* inputDelta_d = new matrix();
    matrix* outputDelta_d = new matrix();

    for(int d=0; d<inputDelta->depth; ++d){
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        outputDelta_d->SetToTensorLayer(outputDelta, d);
        BackwardAveragePool2D_2_2(inputDelta_d, outputDelta_d);
    }

    DeleteOnlyShell(inputDelta_d);
    DeleteOnlyShell(outputDelta_d);
}

void BackwardAveragePool3D_all(tensor* inputDelta, tensor* outputDelta){
    double fact = 1.0/(inputDelta->rows * inputDelta->cols);
    matrix* inputDelta_d = new matrix();
    for(int d=0; d<inputDelta->depth; ++d){
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        inputDelta_d->Add(outputDelta->elem[d] * fact);
    }
    DeleteOnlyShell(inputDelta_d);
}

void BackwardAveragePool3D_all(tensor* inputDelta, tensor* outputDelta, int numNonZero){
    double fact = 1.0/(numNonZero);
    matrix* inputDelta_d = new matrix();
    for(int d=0; d<inputDelta->depth; ++d){
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        inputDelta_d->Add(outputDelta->elem[d] * fact);
    }
    DeleteOnlyShell(inputDelta_d);
}

void BackwardAveragePool3D(tensor* inputDelta, tensor* outputDelta, int kernelR, int kernelC){
    if (kernelR == 2 && kernelC == 2){
        BackwardAveragePool3D_2_2(inputDelta, outputDelta);
        return;
    }
    if (inputDelta->rows == kernelR && inputDelta->cols == kernelC){
        BackwardAveragePool3D_all(inputDelta, outputDelta);
        return;
    }
    matrix* inputDelta_d = new matrix();
    matrix* outputDelta_d = new matrix();

    for(int d=0; d<inputDelta->depth; ++d){
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        outputDelta_d->SetToTensorLayer(outputDelta, d);

        BackwardAveragePool2D(inputDelta_d, outputDelta_d, kernelR, kernelC);
    }

    DeleteOnlyShell(inputDelta_d);
    DeleteOnlyShell(outputDelta_d);
}




void MaxPool2D(matrix* input, matrix* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){

    int minIR, maxIR, minIC, maxIC, outRows = output->rows, outCols = output->cols;
    double *out_oR , *in_iR;
    int *rowInd_oR, *colInd_oR;

    output->SetToValue(-1E10);
    for(int oR=0; oR<outRows; ++oR){
        minIR = oR * kernelRsize;
        maxIR = minIR + kernelRsize - 1;
        out_oR = output->Row(oR);
        rowInd_oR = rowInd + oR*outCols;
        colInd_oR = colInd + oR*outCols;
        for(int iR=minIR; iR<=maxIR; ++iR){
            in_iR = input->Row(iR);
            for(int oC=0; oC<outCols; ++oC){
                minIC = oC*kernelCsize;
                maxIC = minIC + kernelCsize - 1;
                for(int iC=minIC; iC<=maxIC; ++iC){
                    if (out_oR[oC]<in_iR[iC]){
                        out_oR[oC] = in_iR[iC];
                        rowInd_oR[oC] = iR;
                        colInd_oR[oC] = iC;
                    }
                }
                    //out_oR[oC] += in_iR[iC];
            }
        }
    }
}



void BackwardMaxPool2D(matrix* inputDelta, matrix* outputDelta, int* rowInd, int*colInd){
    double *outputDelta_r;
    int *rowInd_r, *colInd_r;
    int outRows = outputDelta->rows, outCols = outputDelta->cols;
    for(int r=0; r<outRows; ++r){
        outputDelta_r = outputDelta->Row(r);
        rowInd_r = rowInd + r*outCols;
        colInd_r = colInd + r*outCols;
        for(int c=0; c<outCols; ++c)
            inputDelta->At(rowInd_r[c], colInd_r[c]) += outputDelta_r[c];
    }
}




void MaxAbsPool2D_2_2(matrix* input, matrix* output, matrix* maxAbs, int* rowInd, int *colInd){

    int minIR, maxIR, minIC, maxIC, outRows = output->rows, outCols = output->cols;
    double *out_oR , *in_iR, *maxAbs_oR;
    int *rowInd_oR, *colInd_oR;
    int iR, iC;

    for(int oR=0; oR<outRows; ++oR){
        out_oR = output->Row(oR);
        maxAbs_oR = maxAbs->Row(oR);
        rowInd_oR = rowInd + oR*outCols;
        colInd_oR = colInd + oR*outCols;

        iR = oR * 2;
        in_iR = input->Row(iR);
        for(int oC=0; oC<outCols; ++oC){
            iC = oC*2;
            maxAbs_oR[oC] = fabs(in_iR[iC]);
            out_oR[oC] = in_iR[iC];
            rowInd_oR[oC] = iR;
            colInd_oR[oC] = iC;

            ++iC;
            if (maxAbs_oR[oC] < fabs(in_iR[iC])){
                maxAbs_oR[oC] = fabs(in_iR[iC]);
                out_oR[oC] = in_iR[iC];
                rowInd_oR[oC] = iR;
                colInd_oR[oC] = iC;
            }
        }

        ++iR;
        in_iR = input->Row(iR);
        for(int oC=0; oC<outCols; ++oC){
            iC = oC*2;
            if (maxAbs_oR[oC] < fabs(in_iR[iC])){
                maxAbs_oR[oC] = fabs(in_iR[iC]);
                out_oR[oC] = in_iR[iC];
                rowInd_oR[oC] = iR;
                colInd_oR[oC] = iC;
            }
            ++iC;
            if (maxAbs_oR[oC] < fabs(in_iR[iC])){
                maxAbs_oR[oC] = fabs(in_iR[iC]);
                out_oR[oC] = in_iR[iC];
                rowInd_oR[oC] = iR;
                colInd_oR[oC] = iC;
            }
        }
    }
}



void MaxAbsPool2D(matrix* input, matrix* output, matrix* maxAbs, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){

    int minIR, maxIR, minIC, maxIC, outRows = output->rows, outCols = output->cols;
    double *out_oR , *in_iR, *maxAbs_oR;
    int *rowInd_oR, *colInd_oR;

    maxAbs->SetToValue(-1);
    for(int oR=0; oR<outRows; ++oR){
        minIR = oR * kernelRsize;
        maxIR = minIR + kernelRsize - 1;

        out_oR = output->Row(oR);
        maxAbs_oR = maxAbs->Row(oR);

        rowInd_oR = rowInd + oR*outCols;
        colInd_oR = colInd + oR*outCols;
        for(int iR=minIR; iR<=maxIR; ++iR){
            in_iR = input->Row(iR);

            for(int oC=0; oC<outCols; ++oC){

                minIC = oC*kernelCsize;
                maxIC = minIC + kernelCsize - 1;

                for(int iC=minIC; iC<=maxIC; ++iC){
                    if (maxAbs_oR[oC] < fabs(in_iR[iC])){
                        maxAbs_oR[oC] = fabs(in_iR[iC]);
                        out_oR[oC] = in_iR[iC];
                        rowInd_oR[oC] = iR;
                        colInd_oR[oC] = iC;
                    }
                }
            }
        }
    }
}


void BackwardMaxAbsPool2D(matrix* inputDelta, matrix* outputDelta, int* rowInd, int*colInd){
    BackwardMaxPool2D(inputDelta, outputDelta, rowInd, colInd);
}



//void MaxPoolSoftMaxIndex2D(matrix* input, double& output, double& softMaxRow, double& softMaxCol,
//                         matrix* softMaxInput, int& rowInd, int& colInd, int kernelRsize, int kernelCsize){
//    output = input->elem[0][0];
//
//    for(int r=0; r<input->rows; ++r)
//        for(int c=0; c<input->cols; ++c){
//            if (input[r][c])
//        }
//}








void MaxPool3D(tensor* input, tensor* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){
    matrix* input_d = new matrix();
    matrix* output_d = new matrix();
    int oRows = output->rows, oCols = output->cols;
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output_d->SetToTensorLayer(output, d);
        MaxPool2D(input_d, output_d, rowInd + d * oRows * oCols, colInd + d * oRows * oCols, kernelRsize, kernelCsize);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(output_d);
}

void BackwardMaxPool3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd){
    matrix* inputDelta_d = new matrix();
    matrix* outputDelta_d = new matrix();
    int oRows = outputDelta->rows, oCols = outputDelta->cols;
    for(int d=0; d<inputDelta->depth; ++d){
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        outputDelta_d->SetToTensorLayer(outputDelta, d);

        BackwardMaxPool2D(inputDelta_d, outputDelta_d, rowInd+d * oRows * oCols, colInd+d * oRows * oCols);
    }
    DeleteOnlyShell(inputDelta_d);
    DeleteOnlyShell(outputDelta_d);
}

void MaxAbsPool3D_2_2(tensor* input, tensor* output, tensor* maxAbs, int* rowInd, int *colInd){
    matrix* input_d = new matrix();
    matrix* output_d = new matrix();
    matrix* maxAbs_d = new matrix();
    int oRows = output->rows, oCols = output->cols;
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output_d->SetToTensorLayer(output, d);
        maxAbs_d->SetToTensorLayer(maxAbs, d);

        MaxAbsPool2D_2_2(input_d, output_d, maxAbs_d, rowInd + d * oRows * oCols, colInd + d * oRows * oCols);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(output_d);
    DeleteOnlyShell(maxAbs_d);
}



void MaxAbsPool3D(tensor* input, tensor* output, tensor* maxAbs, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){
    if (kernelRsize == 2 && kernelCsize == 2){
        MaxAbsPool3D_2_2(input, output, maxAbs, rowInd, colInd);
        return;
    }
    matrix* input_d = new matrix();
    matrix* output_d = new matrix();
    matrix* maxAbs_d = new matrix();
    int oRows = output->rows, oCols = output->cols;
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output_d->SetToTensorLayer(output, d);
        maxAbs_d->SetToTensorLayer(maxAbs, d);

        MaxAbsPool2D(input_d, output_d, maxAbs_d, rowInd + d * oRows * oCols, colInd + d * oRows * oCols, kernelRsize, kernelCsize);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(output_d);
    DeleteOnlyShell(maxAbs_d);
}


void BackwardMaxAbsPool3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd){
    BackwardMaxPool3D(inputDelta, outputDelta, rowInd, colInd);
}

void MaxAbsPoolIndex3D(tensor* input, tensor* output, tensor* maxAbs, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){
    tensor* maxOut = new tensor();
    maxOut->SubTensor(output, input->depth);

    MaxAbsPool3D(input, maxOut, maxAbs, rowInd, colInd, kernelRsize, kernelCsize);

    int startingIndex = maxOut->len;
    double* out_start = output->elem + startingIndex;
    double row0 = rowInd[0], col0 = colInd[0];

    for(int j=0; j<maxOut->len; ++j)
        out_start[j] = (rowInd[j] - row0) / (double) input->rows;

    out_start = output->elem + 2 * startingIndex;
    for(int j=0; j<maxOut->len; ++j)
        out_start[j] = (colInd[j] - col0) / (double) input->cols;

    DeleteOnlyShell(maxOut);
}


void BackwardMaxAbsPoolIndex3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd){
    tensor* outputMaxDelta = new tensor();
    outputMaxDelta->SubTensor(outputDelta, inputDelta->depth);
    BackwardMaxAbsPool3D(inputDelta, outputMaxDelta, rowInd, colInd);
}





void CalculateSoftIndex(matrix* input, matrix* softMaxInput, double & softMaxRow, double & softMaxCol, double & maxVal, double logFactor){
    double sum=0;
    double* input_r, *softMaxInput_r;
    double iR = input->rows, iC = input->cols;
    for(int r=0; r<input->rows; ++r){
        input_r = input->Row(r);
        softMaxInput_r = softMaxInput->Row(r);
        for(int c=0; c<input->cols; ++c){
            softMaxInput_r[c] = exp( logFactor * (fabs(input_r[c]) - maxVal) );
            sum += softMaxInput_r[c];
        }
    }

    softMaxRow = 0.5;
    softMaxCol = 0.5;
    double addon_r;
    double addon_c[input->cols];
    for(int c=0; c<input->cols; ++c)
        addon_c[c]=0;
    for(int r=0; r<input->rows; ++r){
        softMaxInput_r = softMaxInput->Row(r);
        addon_r  = 0;
        for(int c=0; c<input->cols; ++c){
            softMaxInput_r[c] /= sum;
            addon_r += softMaxInput_r[c];
            addon_c[c] += softMaxInput_r[c];
            //softMaxRow += softMaxInput_r[c] * r;
            //softMaxCol += softMaxInput_r[c] * c;
        }
        softMaxRow += addon_r * r;
    }
    for(int c=0; c<input->cols; ++c)
        softMaxCol += addon_c[c] * c;

    softMaxRow /= iR;
    softMaxCol /= iC;
}


void MaxAbsPoolSoftIndex3D(tensor* input, vect* output, vect* maxAbs, tensor* softMaxInput, int* rowInd, int* colInd, double logFactor){
    int iD = input->depth;
    tensor* maxOut = new tensor();
    maxOut->SetSize(iD, 1, 1);
    maxOut->PointToTensor(output->elem);

    tensor* maxAbsTens = new tensor();
    maxAbsTens->SetSize(iD, 1, 1);
    maxAbsTens->PointToTensor(maxAbs->elem);

    MaxAbsPool3D(input, maxOut, maxAbsTens, rowInd, colInd, input->rows, input->cols);

    matrix* input_d = new matrix();
    matrix* softMaxInput_d = new matrix();

    for(int d=0; d<iD; ++d){
        input_d->SetToTensorLayer(input, d);
        softMaxInput_d->SetToTensorLayer(softMaxInput, d);
        CalculateSoftIndex(input_d, softMaxInput_d, output->elem[iD + d], output->elem[2 * iD + d], maxAbs->elem[d], logFactor);
    }

    DeleteOnlyShell(input_d);
    DeleteOnlyShell(softMaxInput_d);
    DeleteOnlyShell(maxOut);
    DeleteOnlyShell(maxAbsTens);
}


void BackwardSoftIndex(matrix* input, matrix* softMaxInput, const double softRow, const double softCol, matrix* inputDelta,
                       const double deltaSoftRow, const double deltaSoftCol, double logFactor){
    double *inputDelta_r, *softMaxInput_r, *input_r;
    double iR = input->rows, iC = input->cols;
    double tempR, coefC;

    coefC = deltaSoftCol / iC * logFactor;

    for(int r=0; r<input->rows; ++r){
        inputDelta_r = inputDelta->Row(r);
        softMaxInput_r = softMaxInput->Row(r);
        input_r = input->Row(r);
        tempR = logFactor * deltaSoftRow * ( double(r) / iR - softRow) - deltaSoftCol * softCol;

        for(int c=0; c<input->cols; ++c){
            //inputDelta_r[c] += logFactor * sign(input_r[c]) * softMaxInput_r[c] * (deltaSoftRow * ( double(r) / iR - softRow) + deltaSoftCol * (double(c) / iC - softCol) );
            inputDelta_r[c] += sign(input_r[c]) * softMaxInput_r[c] * (tempR + coefC * c);
        }
    }
}


void BackwardMaxAbsPoolSoftIndex3D(tensor* input, tensor* softMaxInput, vect* output, tensor* inputDelta, vect* outputDelta, int* rowInd, int* colInd, double logFactor){
    int iD = input->depth;
    tensor* maxOutDelta = new tensor();
    maxOutDelta->SetSize(iD, 1, 1);
    maxOutDelta->PointToTensor(outputDelta->elem);

    BackwardMaxAbsPool3D(inputDelta, maxOutDelta, rowInd, colInd);

    matrix* input_d = new matrix();
    matrix* softMaxInput_d = new matrix();
    matrix* inputDelta_d = new matrix();

    for(int d=0; d<iD; ++d){
        input_d->SetToTensorLayer(input, d);
        softMaxInput_d->SetToTensorLayer(softMaxInput, d);
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        BackwardSoftIndex(input_d, softMaxInput_d, output->elem[d + iD], output->elem[d + 2 * iD], inputDelta_d, outputDelta->elem[d + iD], outputDelta->elem[d + 2 * iD], logFactor);
    }

    DeleteOnlyShell(maxOutDelta);
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(softMaxInput_d);
    DeleteOnlyShell(inputDelta_d);

}




void MaxAbsPoolSoftDiffIndex3D(tensor* input, vect* output, vect* tempOutput, vect* maxAbs, tensor* softMaxInput, int* rowInd, int* colInd, double logFactor){
    tempOutput->SetToZero();
    MaxAbsPoolSoftIndex3D(input, tempOutput, maxAbs, softMaxInput, rowInd, colInd, logFactor);
    output->Copy(tempOutput);

    int iD = input->depth;
    double softRow0 = output->elem[iD];
    double softCol0 = output->elem[2 * iD];

    for(int d=0; d<input->depth; ++d){
        output->elem[d + iD] -= softRow0;
        output->elem[d + 2 * iD] -= softCol0;
    }
}

void BackwardMaxAbsPoolSoftDiffIndex3D(tensor* input, tensor* softMaxInput, vect* tempOutput, tensor* inputDelta, vect* outputDelta, vect* tempOutputDelta,
                                   int* rowInd, int* colInd, double logFactor){
    tempOutputDelta->Copy(outputDelta);

    int iD = input->depth;
    double rowDeltaSum=0, colDeltaSum=0;
    for(int d=1; d<input->depth; ++d){
        rowDeltaSum += outputDelta->elem[d + iD];
        colDeltaSum += outputDelta->elem[d + 2 * iD];
    }
    tempOutputDelta->elem[iD]     = - rowDeltaSum;
    tempOutputDelta->elem[2 * iD] = - colDeltaSum;

    BackwardMaxAbsPoolSoftIndex3D(input, softMaxInput, tempOutput, inputDelta, tempOutputDelta, rowInd, colInd, logFactor);
}

//void MaxPoolSoftMaxIndex3D(tensor* input, vect* output, vect* softMaxRow, vect* softMaxCol,
//                         tensor* softMaxInput, int* rowInd, int* colInd, int kernelRsize, int kernelCsize){
//    matrix* input_d = new matrix();
//    realNumber* output_d = new realNumber();
//    realNumber* softMaxRow_d = new realNumber();
//    realNumber* softMaxCol_d = new realNumber();
//    matrix* softMaxInput_d = new matrix();
//
//    for(int d=0; d<input->depth; ++d){
//        input_d->SetToTensorLayer(input, d);
//        output_d->SetToVectElement(output, d);
//        softMaxRow_d->SetToVectElement(softMaxRow, d);
//        softMaxCol_d->SetToVectElement(softMaxCol, d);
//        softMaxInput_d->SetToTensorLayer(softMaxInput, d);
//
//        MaxPoolSoftMaxIndex2D(input_d, output->elem[d], softMaxRow->elem[d], softMaxCol->elem[d], softMaxInput_d, rowInd[d], colInd[d], kernelRsize, kernelCsize);
//
//    }
//
//
//    DeleteOnlyShell(input_d);
//    DeleteOnlyShell(output_d);
//    DeleteOnlyShell(softMaxRow_d);
//    DeleteOnlyShell(softMaxCol_d);
//    DeleteOnlyShell(softMaxInput_d);
//
//}



void MinPool2D(matrix* input, matrix* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){

    int minIR, maxIR;
    int minIC, maxIC;
    output->SetToValue(1E10);
    double *out_oR , *in_iR;
    int outRows = output->rows, outCols = output->cols;
    int *rowInd_oR, *colInd_oR;
    for(int oR=0; oR<outRows; ++oR){
        minIR = oR * kernelRsize;
        maxIR = minIR + kernelRsize - 1;
        out_oR = output->Row(oR);
        rowInd_oR = rowInd + oR*outCols;
        colInd_oR = colInd + oR*outCols;
        for(int iR=minIR; iR<=maxIR; ++iR){

            in_iR = input->Row(iR);

            for(int oC=0; oC<outCols; ++oC){
                minIC = oC*kernelCsize;
                maxIC = minIC + kernelCsize - 1;
                for(int iC=minIC; iC<=maxIC; ++iC){
                    if (out_oR[oC]>in_iR[iC]){
                        out_oR[oC] = in_iR[iC];
                        rowInd_oR[oC] = iR;
                        colInd_oR[oC] = iC;
                    }
                }
                    //out_oR[oC] += in_iR[iC];
            }
        }
    }
}


void BackwardMinPool2D(matrix* inputDelta, matrix* outputDelta, int* rowInd, int*colInd){
    double *outputDelta_r;
    int *rowInd_r, *colInd_r;
    int outRows = outputDelta->rows, outCols = outputDelta->cols;
    for(int r=0; r<outRows; r++){
        outputDelta_r = outputDelta->Row(r);
        rowInd_r = rowInd + r*outCols;
        colInd_r = colInd + r*outCols;
        for(int c=0; c<outCols; c++)
            inputDelta->At(rowInd_r[c], colInd_r[c]) += outputDelta_r[c];
    }
}


void MinPool3D(tensor* input, tensor* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize){
    matrix* input_d = new matrix();
    matrix* output_d = new matrix();
    int oRows = output->rows, oCols = output->cols;
    for(int d=0; d<input->depth; ++d){
        input_d->SetToTensorLayer(input, d);
        output_d->SetToTensorLayer(output, d);
        MinPool2D(input_d, output_d, rowInd + d * oRows * oCols, colInd + d * oRows * oCols, kernelRsize, kernelCsize);
    }
    DeleteOnlyShell(input_d);
    DeleteOnlyShell(output_d);
}

void BackwardMinPool3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd){
    matrix* inputDelta_d = new matrix();
    matrix* outputDelta_d = new matrix();
    int oRows = outputDelta->rows, oCols = outputDelta->cols;
    for(int d=0; d<inputDelta->depth; ++d){
        inputDelta_d->SetToTensorLayer(inputDelta, d);
        outputDelta_d->SetToTensorLayer(outputDelta, d);
        BackwardMinPool2D(inputDelta_d, outputDelta_d, rowInd+d * oRows * oCols, colInd+d * oRows * oCols);
    }
    DeleteOnlyShell(inputDelta_d);
    DeleteOnlyShell(outputDelta_d);
}

void FillRandom(int* index, int startInd, int endInd, int len){
    int tempArray[endInd - startInd];
    for(int j=startInd; j<endInd; ++j)
        tempArray[j-startInd] = j;
    int randInd;
    int tempInd;
    for(int j=0; j<len; ++j){
        randInd = (randomGenerator::rand() % (endInd - (startInd + j) ) ) + j;
        Switch(tempArray[j], tempArray[randInd]);
        index[j] = tempArray[j];
    }
    sort(index, index + len);
}

void Switch(int & a, int & b){
    int temp = a;
    a = b;
    b = temp;
}

void MeanVarPool(orderedData* input, double * mean, double * var){
    double mean_ = 0, meanQuad_ = 0;
    for(int j=0; j<input->len; ++j){
        mean_ += input->elem[j];
        meanQuad_ += sqr(input->elem[j]);
    }
    * mean = mean_ / input->len;
    * var = meanQuad_ / input->len - sqr(*mean);
}

void MeanVarPoolTensor(tensor* input, tensor* output){
    matrix* input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        MeanVarPool(input_j, output->elem + 2 * j, output->elem + 2 * j + 1);
    }
    DeleteOnlyShell(input_j);
}

void BackwardMeanVarPool(orderedData* input, double * mean, double * var, orderedData* inputDelta, double * meanDelta, double * varDelta){
    double a = (*meanDelta) / input->len - (*varDelta) * 2.0 / input->len * (*mean);
    double b = (*varDelta) / input->len * 2.0;
    for(int j=0; j<input->len; ++j){
        inputDelta->elem[j] += a + b * input->elem[j];
    }
}

void BackwardMeanVarPoolTensor(tensor* input, tensor* output, tensor* inputDelta, tensor* outputDelta){
    matrix * input_j = new matrix();
    matrix* inputDelta_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        BackwardMeanVarPool(input_j, output->elem + 2 * j, output->elem + 2 * j + 1, inputDelta_j, outputDelta->elem + 2 * j, outputDelta->elem + 2 * j + 1);
    }
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
}






void MeanQuadStatsPool(matrix* input, double * stats){
    double mean = 0, var = 0, covHor = 0, covVert = 0;
    for(int j=0; j<input->len; ++j){
        mean += input->elem[j];
        var += sqr(input->elem[j]);
    }
    mean /= input->len;
    var /= input->len;
    //var -= sqr(mean);

    double * input_r;
    for(int r=0; r<input->rows; ++r){
        input_r = input->Row(r);
        for(int c=0; c<input->cols - 1; ++c)
            covHor += input_r[c] * input_r[c+1];
    }
    covHor /= input->rows * (input->cols - 1);
    //covHor -= sqr(mean);

    double * input_r1;
    for(int r=0; r<input->rows - 1; ++r){
        input_r = input->Row(r);
        input_r1 = input->Row(r+1);
        for(int c=0; c<input->cols; ++c)
            covVert += input_r[c] * input_r1[c];
    }
    covVert /= (input->rows - 1) * input->cols;
    //covVert -= sqr(mean);

    stats[0] = mean;
    stats[1] = var;
    stats[2] = covHor;
    stats[3] = covVert;
}


void MeanQuadStatsPoolTensor(tensor* input, tensor* output){
    matrix* input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        MeanQuadStatsPool(input_j, output->elem + 4 * j);
    }
    DeleteOnlyShell(input_j);
}


void BackwardMeanQuadStatsPool(matrix* input, double * stats, matrix* inputDelta, double * statsDelta){
    double a = (statsDelta[0]) / input->len;// - 2.0 * stats[0] / input->len * (statsDelta[1] + statsDelta[2] + statsDelta[3]);
    double b = (statsDelta[1]) / input->len * 2.0;
    for(int j=0; j<input->len; ++j){
        inputDelta->elem[j] += a + b * input->elem[j];
    }

    double * input_r, * input_r1;
    double * inputDelta_r, * inputDelta_r1;
    double mult = statsDelta[2] / (input->rows * (input->cols - 1));
    for(int r=0; r<input->rows; ++r){
        input_r = input->Row(r);
        inputDelta_r = inputDelta->Row(r);
        for(int c=0; c<input->cols - 1; ++c){
            inputDelta_r[c] += input_r[c+1] * mult;
            inputDelta_r[c+1] += input_r[c] * mult;
        }
    }

    mult = statsDelta[3] / ((input->rows - 1) * input->cols);
    for(int r=0; r<input->rows - 1; ++r){
        input_r = input->Row(r);
        input_r1 = input->Row(r+1);
        inputDelta_r = inputDelta->Row(r);
        inputDelta_r1 = inputDelta->Row(r+1);
        for(int c=0; c<input->cols; ++c){
            inputDelta_r[c] += input_r1[c] * mult;
            inputDelta_r1[c] += input_r[c] * mult;
        }
    }
}


void BackwardMeanQuadStatsPoolTensor(tensor* input, tensor* output, tensor* inputDelta, tensor* outputDelta){
    matrix * input_j = new matrix();
    matrix* inputDelta_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        BackwardMeanQuadStatsPool(input_j, output->elem + 4 * j, inputDelta_j, outputDelta->elem + 4 * j);
    }
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
}








void MeanStDevPool(orderedData* input, double * mean, double * stDev){
    double mean_ = 0, meanQuad_ = 0;
    for(int j=0; j<input->len; ++j){
        mean_ += input->elem[j];
        meanQuad_ += sqr(input->elem[j]);
    }
    * mean = mean_ / input->len;
    * stDev = meanQuad_ / input->len - sqr(*mean);
    *stDev = sqrt(*stDev);
}

void MeanStDevPoolTensor(tensor* input, tensor* output){
    matrix* input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        MeanStDevPool(input_j, output->elem + 2 * j, output->elem + 2 * j + 1);
    }
    DeleteOnlyShell(input_j);
}

void BackwardMeanStDevPool(orderedData* input, double * mean, double * stDev, orderedData* inputDelta, double * meanDelta, double * stDevDelta){
    double a = (*meanDelta) / input->len;
    double b = (*stDevDelta) / (input->len * (*stDev));
    for(int j=0; j<input->len; ++j){
        inputDelta->elem[j] += a + b * (input->elem[j] - (*mean));
    }
}

void BackwardMeanStDevPoolTensor(tensor* input, tensor* output, tensor* inputDelta, tensor* outputDelta){
    matrix * input_j = new matrix();
    matrix* inputDelta_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        BackwardMeanStDevPool(input_j, output->elem + 2 * j, output->elem + 2 * j + 1, inputDelta_j, outputDelta->elem + 2 * j, outputDelta->elem + 2 * j + 1);
    }
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(inputDelta_j);
}









void AverageMaxAbsPoolMatrixAll(matrix* input, double * stats, int * rowInd, int * colInd){
    double maxAbs = -1, sum = 0;
    double * input_r;
    for(int r=0; r<input->rows; ++r){
        input_r = input->Row(r);
        for(int c=0; c<input->cols; ++c){
            sum += input_r[c];
            if (fabs(input_r[c]) > maxAbs){
                maxAbs = fabs(input_r[c]);
                stats[1] = input_r[c];
                rowInd[0] = r;
                colInd[0] = c;
            }
        }
    }
    stats[0] = sum / input->len;
}

void AverageMaxAbsPoolMatrix_2_2(matrix* input, matrix* outputAverage, matrix* outputMaxAbs, int * rowInd, int * colInd){
    double * input_2r, * input_2r_1, * outputAverage_r, * outputMaxAbs_r;
    int * rowInd_r, * colInd_r;
    double maxAbs;


    for(int r = 0; r < outputAverage->rows; ++r){
        input_2r = input->Row(2 * r);
        input_2r_1 = input->Row(2 * r + 1);
        outputAverage_r = outputAverage->Row(r);
        outputMaxAbs_r = outputMaxAbs->Row(r);
        rowInd_r = rowInd + r * outputMaxAbs->cols;
        colInd_r = colInd + r * outputMaxAbs->cols;
        for(int c=0; c<outputAverage->cols; ++c){
            outputAverage_r[c] += input_2r [2 * c] + input_2r [2 * c + 1] + input_2r_1 [2 * c] + input_2r_1 [2 * c + 1];

            maxAbs =        fabs(input_2r[2 * c]);
            outputMaxAbs_r[c] =  input_2r[2 * c];
            rowInd_r[c] = 2 * r;
            colInd_r[c] = 2 * c;

            if (fabs(input_2r[2 * c + 1]) > maxAbs){
                maxAbs =        fabs(input_2r[2 * c + 1]);
                outputMaxAbs_r[c] =  input_2r[2 * c + 1];
                rowInd_r[c] = 2 * r;
                colInd_r[c] = 2 * c + 1;
            }

            if (fabs(input_2r_1[2 * c]) > maxAbs){
                maxAbs =        fabs(input_2r_1[2 * c]);
                outputMaxAbs_r[c] =  input_2r_1[2 * c];
                rowInd_r[c] = 2 * r + 1;
                colInd_r[c] = 2 * c;
            }

            if (fabs(input_2r_1[2 * c + 1]) > maxAbs){
                maxAbs =        fabs(input_2r_1[2 * c + 1]);
                outputMaxAbs_r[c] =  input_2r_1[2 * c + 1];
                rowInd_r[c] = 2 * r + 1;
                colInd_r[c] = 2 * c + 1;
            }
        }
    }

    outputAverage->Multiply(0.25);
}




void AverageMaxAbsPool(tensor* input, tensor* output, int kernelRsize, int kernelCsize, int * rowInd, int * colInd, int onlyAveragePoolingDepth){
    tensor* inputOnlyAverage = new tensor();
    inputOnlyAverage->SubTensor(input, onlyAveragePoolingDepth);

    tensor* outOnlyAverage = new tensor();
    outOnlyAverage->SubTensor(output, onlyAveragePoolingDepth);

    AveragePool3D(inputOnlyAverage, outOnlyAverage, kernelRsize, kernelCsize);

    matrix* input_j = new matrix();
    matrix* outputAverage_j = new matrix();
    matrix* outputMaxAbs_j = new matrix();

    int outMatrixSize = output->rows * output->cols;
    int indShift_j;

    if (kernelRsize == input->rows && kernelCsize == input->cols){
        for(int j = onlyAveragePoolingDepth; j < input->depth; ++j){
            input_j->SetToTensorLayer(input, j);
            indShift_j = (j - onlyAveragePoolingDepth) * outMatrixSize;
            AverageMaxAbsPoolMatrixAll(input_j, output->elem + 2 * j - onlyAveragePoolingDepth, rowInd + indShift_j, colInd + indShift_j);
        }
    }

    else{
        if (kernelRsize == 2 && kernelCsize == 2)
        {

            for(int j = onlyAveragePoolingDepth; j < input->depth; ++j){
                input_j->SetToTensorLayer(input, j);
                outputAverage_j->SetToTensorLayer(output, 2 * j - onlyAveragePoolingDepth);
                outputMaxAbs_j->SetToTensorLayer(output, 2 * j - onlyAveragePoolingDepth + 1);
                indShift_j = (j - onlyAveragePoolingDepth) * outMatrixSize;
                AverageMaxAbsPoolMatrix_2_2(input_j, outputAverage_j, outputMaxAbs_j, rowInd + indShift_j, colInd + indShift_j);
            }
        }

        else{
            cout<<"General size Average Max Abs Pool is not implemented yet"<<endl;
        }
    }

    DeleteOnlyShell(inputOnlyAverage);
    DeleteOnlyShell(outOnlyAverage);
    DeleteOnlyShell(input_j);
    DeleteOnlyShell(outputAverage_j);
    DeleteOnlyShell(outputMaxAbs_j);
}



void BackwardAverageMaxAbsPool(tensor* inputDelta, tensor* outputDelta,
                               int kernelRsize, int kernelCsize, int * rowInd, int * colInd, int onlyAveragePoolingDepth){
    tensor* inputDeltaOnlyAverage = new tensor();
    inputDeltaOnlyAverage->SubTensor(inputDelta, onlyAveragePoolingDepth);

    tensor* outputDeltaOnlyAverage = new tensor();
    outputDeltaOnlyAverage->SubTensor(outputDelta, onlyAveragePoolingDepth);

    BackwardAveragePool3D(inputDeltaOnlyAverage, outputDeltaOnlyAverage, kernelRsize, kernelCsize);

    matrix* inputDelta_j = new matrix();
    matrix* outputDeltaAverage_j = new matrix();
    matrix* outputDeltaMaxAbs_j = new matrix();
    int indShift_j;
    int outMatrixSize = outputDelta->rows * outputDelta->cols;

    if (kernelRsize == inputDelta->rows && kernelCsize == inputDelta->cols){
        double fact = 1.0 / (inputDelta->rows * inputDelta->cols);
        for(int j = onlyAveragePoolingDepth; j < inputDelta->depth; ++j){
            inputDelta_j->SetToTensorLayer(inputDelta, j);
            indShift_j = (j - onlyAveragePoolingDepth);
            inputDelta_j->Add(outputDelta->elem[2 * j - onlyAveragePoolingDepth] * fact);
            inputDelta_j->At(rowInd[indShift_j], colInd[indShift_j]) += outputDelta->elem[2 * j - onlyAveragePoolingDepth + 1];
        }
    }

    else{
        if (kernelRsize == 2 && kernelCsize == 2)
        {
            for(int j = onlyAveragePoolingDepth; j < inputDelta->depth; ++j){
                inputDelta_j->SetToTensorLayer(inputDelta, j);
                outputDeltaAverage_j->SetToTensorLayer(outputDelta, 2 * j - onlyAveragePoolingDepth);
                outputDeltaMaxAbs_j->SetToTensorLayer(outputDelta, 2 * j - onlyAveragePoolingDepth + 1);
                indShift_j = (j - onlyAveragePoolingDepth) * outMatrixSize;
                BackwardAveragePool2D_2_2(inputDelta_j, outputDeltaAverage_j);
                BackwardMaxAbsPool2D(inputDelta_j, outputDeltaMaxAbs_j, rowInd + indShift_j, colInd + indShift_j);
            }
        }
        else
        {
            for(int j = onlyAveragePoolingDepth; j < inputDelta->depth; ++j){
                inputDelta_j->SetToTensorLayer(inputDelta, j);
                outputDeltaAverage_j->SetToTensorLayer(outputDelta, 2 * j - onlyAveragePoolingDepth);
                outputDeltaMaxAbs_j->SetToTensorLayer(outputDelta, 2 * j - onlyAveragePoolingDepth + 1);
                indShift_j = (j - onlyAveragePoolingDepth) * outMatrixSize;
                BackwardAveragePool2D(inputDelta_j, outputDeltaAverage_j, kernelRsize, kernelCsize);
                BackwardMaxAbsPool2D(inputDelta_j, outputDeltaMaxAbs_j, rowInd + indShift_j, colInd + indShift_j);
            }
        }

    }

    DeleteOnlyShell(inputDeltaOnlyAverage);
    DeleteOnlyShell(outputDeltaOnlyAverage);
    DeleteOnlyShell(inputDelta_j);
    DeleteOnlyShell(outputDeltaAverage_j);
    DeleteOnlyShell(outputDeltaMaxAbs_j);
}


void CenterPool(matrix* input, double * centers, double * sum){
    double sum_0 = 0;
    double sum_r = 0, sum_c = 0;
    double * input_r;
    for(int r=0; r<input->rows; ++r){
        input_r = input->Row(r);
        for(int c=0; c<input->cols; ++c){
            sum_0 += input_r[c];
            sum_r += r * input_r[c];
            sum_c += c * input_r[c];
        }
    }
    sum_0 += 1E-5;
    centers[0] = sum_r / (sum_0 * input->rows);
    centers[1] = sum_c / (sum_0 * input->cols);
    sum[0] = sum_0;
}

void CenterPoolTensor(tensor* input, tensor* output, vect* sum){
    matrix * input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        CenterPool(input_j, output->elem + 2 * j, sum->elem + j);
    }
    DeleteOnlyShell(input_j);
}

void BackwardCenterPool(double * centers, double * sum, matrix * inputDelta, double * centersDelta){
    double * inputDelta_r;
    double row_addon;
    double col_addon[inputDelta->cols];
    for(int c = 0; c<inputDelta->cols; ++c)
        col_addon[c] = centersDelta[1] * (double(c) / double(inputDelta->cols) - centers[1]) / sum[0];

    for(int r=0; r<inputDelta->rows; ++r){
        inputDelta_r = inputDelta->Row(r);
        row_addon = centersDelta[0] * (double(r) / double(inputDelta->rows) - centers[0]) / sum[0];
        for(int c=0; c<inputDelta->cols; ++c)
            inputDelta_r[c] += row_addon + col_addon[c];
    }
}



void BackwardCenterPoolTensor(tensor * output, vect * sum, tensor * inputDelta, tensor * outputDelta){
    matrix* inputDelta_j = new matrix();

    for(int j=0; j<inputDelta->depth; ++j){
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        BackwardCenterPool(output->elem + 2 * j, sum->elem + j, inputDelta_j, outputDelta->elem + 2 * j);
    }

    DeleteOnlyShell(inputDelta_j);
}



void BackwardMedianPoolTensor(tensor * inputDelta, tensor * outputDelta, int * index){
    for(int j=0; j<inputDelta->depth; ++j)
        inputDelta->elem[inputDelta->Ind(j) + index[j] ] += outputDelta->elem[j];
}

void BackwardMedianNonzeroPoolTensor(tensor* output, tensor * inputDelta, tensor * outputDelta, int * index){
    for(int j=0; j<inputDelta->depth; ++j)
        if (fabs(output->elem[j]) > 1E-8)
            inputDelta->elem[inputDelta->Ind(j) + index[j] ] += outputDelta->elem[j];
}



void MedianPoolTensor(tensor * input, tensor * output, int * index){
    matrix * input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        input_j->computeMedian(output->elem + j, index + j);
    }
    DeleteOnlyShell(input_j);
}


void MedianNonzeroPoolTensor(tensor * input, tensor * output, int * index){
    matrix * input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        input_j->computeMedianNonzero(output->elem + j, index + j);
    }
    DeleteOnlyShell(input_j);
}




void QuartilesPoolTensor(tensor * input, tensor * output, int * index, int numQuartiles){
    matrix * input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        input_j->computeQuartiles(output->elem + numQuartiles * j, index + numQuartiles * j, numQuartiles);
    }
    DeleteOnlyShell(input_j);
}

void QuartilesNonzeroPoolTensor(tensor * input, tensor * output, int * index, int numQuartiles){
    matrix * input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        input_j->computeQuartilesNonzero(output->elem + numQuartiles * j, index + numQuartiles * j, numQuartiles);
    }
    DeleteOnlyShell(input_j);
}

void BackwardQuartilesPoolTensor(tensor * inputDelta, tensor * outputDelta, int * index, int numQuartiles){
    matrix * inputDelta_j = new matrix();
    int * index_j;
    double * outputDelta_j;
    for(int j=0; j<inputDelta->depth; ++j){
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        index_j = index + j * numQuartiles;
        outputDelta_j = outputDelta->elem + j * numQuartiles;
        for(int quart=0; quart<numQuartiles; ++quart){
            inputDelta_j->elem[index_j[quart] ] += outputDelta_j[quart];
        }
    }
    DeleteOnlyShell(inputDelta_j);
}

void BackwardQuartilesNonzeroPoolTensor(tensor * output, tensor * inputDelta, tensor * outputDelta, int * index, int numQuartiles){
    matrix * inputDelta_j = new matrix();
    int * index_j;
    double * outputDelta_j;
    double * output_j;
    for(int j=0; j<inputDelta->depth; ++j){
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        index_j = index + j * numQuartiles;
        outputDelta_j = outputDelta->elem + j * numQuartiles;
        output_j = output->elem + j * numQuartiles;
        for(int quart=0; quart<numQuartiles; ++quart){
            if (fabs(output_j[quart]) > 1E-8)
                inputDelta_j->elem[index_j[quart] ] += outputDelta_j[quart];
        }
    }
    DeleteOnlyShell(inputDelta_j);
}



void FullSubAveragePool2D(matrix* input, double* output, int border){
    double fact = 1.0 / ( (input->rows - 2 * border) * (input->cols - 2 * border) );
    double * input_r;
    double sum = 0;
    for(int r = border; r < input->rows - border; ++r){
        input_r = input->Row(r);
        for(int c = border; c<input->cols - border; ++c)
            sum += input_r[c];
    }
    output[0] = sum * fact;
}



void FullSubAveragePool(tensor* input, tensor* output, int border){
    matrix* input_j = new matrix();
    for(int j=0; j<input->depth; ++j){
        input_j->SetToTensorLayer(input, j);
        FullSubAveragePool2D(input_j, output->elem + j, border);
    }
    DeleteOnlyShell(input_j);
}

void BackwardFullSubAveragePool2D(matrix* inputDelta, double * outputDelta, int border){
    double fact = 1.0 / ( (inputDelta->rows - 2 * border) * (inputDelta->cols - 2 * border) );
    double * inputDelta_r;
    double addon = outputDelta[0] * fact;
    for(int r = border; r < inputDelta->rows - border; ++r){
        inputDelta_r = inputDelta->Row(r);
        for(int c = border; c < inputDelta->cols - border; ++c)
            inputDelta_r[c] += addon;
    }
}

void BackwardFullSubAveragePool(tensor* inputDelta, tensor* outputDelta, int border){
    matrix* inputDelta_j = new matrix();
    for(int j=0; j<inputDelta->depth; ++j){
        inputDelta_j->SetToTensorLayer(inputDelta, j);
        BackwardFullSubAveragePool2D(inputDelta_j, outputDelta->elem + j, border);
    }
    DeleteOnlyShell(inputDelta_j);
}



//Forward pass forward style


//
//
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                output->elem[oR][oC]+=*bias;
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                        output->elem[oR][oC]+=input->elem[iR+kR][iC+kC]*kernel->elem[kR][kC];
//        }


//
//void Convolute2D2D(matrix* input, matrix* output, matrix* kernel, double* bias, int paddingR, int paddingC, int strideR, int strideC){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                output->elem[oR][oC]+=*bias;
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                        output->elem[oR][oC]+=input->elem[iR+kR][iC+kC]*kernel->elem[kR][kC];
//        }
//}
//
//
//void Convolute2D2D(matrix* input, matrix* output, matrix* kernel, double* bias, int paddingR, int paddingC, int strideR, int strideC){
//    for(int oR=0; oR<output->rows; oR++){
//        for(int iR=)
//    }
//
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR){
//        for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//
//
//
//    }
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                output->elem[oR][oC]+=*bias;
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                        output->elem[oR][oC]+=input->elem[iR+kR][iC+kC]*kernel->elem[kR][kC];
//        }
//}
//
//
//void DroppedConvolute2D2D(matrix* input, matrix* output, matrix* kernel, double* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (!activeOutput[oR][oC]) continue;
//                output->elem[oR][oC]+=*bias;
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                        if (activeInput[iR+kR][iC+kC])
//                            output->elem[oR][oC]+=input->elem[iR+kR][iC+kC]*kernel->elem[kR][kC];
//        }
//}
//
//
//
//void BackwardConvolute2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//        for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//            for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++){
//                        inputDelta->elem[iR+kR][iC+kC]+=kernel->elem[kR][kC]*outputDelta->elem[oR][oC];
//                        kernelGrad->elem[kR][kC]+=outputDelta->elem[oR][oC]*input->elem[iR+kR][iC+kC];
//                    }
//            }
//}
//
//void DroppedBackwardConvolute2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput){
//        for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//            for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10 || !activeOutput[oR][oC]) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++){
//                        if (!activeInput[iR+kR][iC+kC]) continue;
//                        inputDelta->elem[iR+kR][iC+kC]+=kernel->elem[kR][kC]*outputDelta->elem[oR][oC];
//                        kernelGrad->elem[kR][kC]+=outputDelta->elem[oR][oC]*input->elem[iR+kR][iC+kC];
//                    }
//            }
//}
//
//void BackwardConvoluteGrad2D2D(matrix* input, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//        for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//            for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                        kernelGrad->elem[kR][kC]+=outputDelta->elem[oR][oC]*input->elem[iR+kR][iC+kC];
//            }
//}
//
//void DroppedBackwardConvoluteGrad2D2D(matrix* input, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput){
//        for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//            for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10 || !activeOutput[oR][oC]) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                    for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                        if (activeInput[iR+kR][iC+kC])
//                            kernelGrad->elem[kR][kC]+=outputDelta->elem[oR][oC]*input->elem[iR+kR][iC+kC];
//            }
//}
//
//
//
//
//void Convolute2D3D(matrix* input, tensor* output, tensor* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC){
//    for(int d=0; d<kernel->depth; d++)
//        Convolute2D2D(input,&(output->elem[d]), &(kernel->elem[d]), &(bias->elem[d]), paddingR, paddingC,strideR, strideC);
//}
//
//void DroppedConvolute2D3D(matrix* input, tensor* output, tensor* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<bool> > &activeInput, vector<vector<vector<bool> > > &activeOutput){
//    for(int d=0; d<kernel->depth; d++)
//        DroppedConvolute2D2D(input,&(output->elem[d]), &(kernel->elem[d]), &(bias->elem[d]), paddingR, paddingC,strideR, strideC, activeInput, activeOutput[d]);
//}
//
//void BackwardConvolute2D3D(matrix* input, matrix* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//    for(int d=0; d<kernel->depth; d++)
//        BackwardConvolute2D2D(input, inputDelta, &(outputDelta->elem[d]), &(kernel->elem[d]), &(kernelGrad->elem[d]), &(bias->elem[d]),
//                              &(biasGrad->elem[d]), paddingR, paddingC, strideR, strideC);
//}
//
//void DroppedBackwardConvolute2D3D(matrix* input, matrix* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<vector<bool> > >&activeOutput){
//    for(int d=0; d<kernel->depth; d++)
//        DroppedBackwardConvolute2D2D(input, inputDelta, &(outputDelta->elem[d]), &(kernel->elem[d]), &(kernelGrad->elem[d]), &(bias->elem[d]),
//                              &(biasGrad->elem[d]), paddingR, paddingC, strideR, strideC, activeInput, activeOutput[d]);
//}
//
//void BackwardConvoluteGrad2D3D(matrix* input, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//    for(int d=0; d<kernel->depth; d++)
//        BackwardConvoluteGrad2D2D(input, &(outputDelta->elem[d]), &(kernel->elem[d]), &(kernelGrad->elem[d]), &(bias->elem[d]),
//                              &(biasGrad->elem[d]), paddingR, paddingC, strideR, strideC);
//}
//
//void DroppedBackwardConvoluteGrad2D3D(matrix* input, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<vector<bool> > >&activeOutput){
//    for(int d=0; d<kernel->depth; d++)
//        DroppedBackwardConvoluteGrad2D2D(input, &(outputDelta->elem[d]), &(kernel->elem[d]), &(kernelGrad->elem[d]), &(bias->elem[d]),
//                              &(biasGrad->elem[d]), paddingR, paddingC, strideR, strideC, activeInput, activeOutput[d]);
//}
//
//
//
//void Convolute3D2D(tensor* input, matrix* output, tensor* kernel, double* bias, int paddingR, int paddingC, int strideR, int strideC){
//    double temp=0;
//    double ** inputElemkD;
//    double ** kernelElemkD;
//    int kRmin, kRmax;
//    int kCmin, kCmax;
//
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                temp = *bias;
//
//                for(int kD=0; kD<kernel->depth; kD++){
//                            inputElemkD = input->elem[kD].elem;
//                            kernelElemkD = kernel->elem[kD].elem;
//                            kRmin = max(0, -iR); kRmax = min(kernel->rows, input->rows-iR);
//                            kCmin = max(0, -iC); kCmax = min(kernel->cols, input->cols-iC);
//
//                    for(int kR=kRmin; kR<kRmax; kR++)
//                        for(int kC=kCmin; kC<kCmax; kC++)
//                            temp+=inputElemkD[iR+kR][iC+kC]*kernelElemkD[kR][kC];
//                }
//
//                output->elem[oR][oC]+=temp;
//        }
//}
//
//
//
//void DroppedConvolute3D2D(tensor* input, matrix* output, tensor* kernel, double* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<vector<bool> > > &activeInput, vector<vector<bool> > &activeOutput){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (!activeOutput[oR][oC]) continue;
//                output->elem[oR][oC]+= *bias;
//                for(int kD=0; kD<kernel->depth; kD++)
//                    for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                        for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                            if (activeInput[kD][iR+kR][iC+kC])
//                                output->elem[oR][oC]+=input->elem[kD][iR+kR][iC+kC]*kernel->elem[kD][kR][kC];
//        }
//}
//
//
//
//void BackwardConvolute3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kD=0; kD<kernel->depth; kD++)
//                    for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                        for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++){
//                                inputDelta->elem[kD][iR+kR][iC+kC]+=kernel->elem[kD][kR][kC]*outputDelta->elem[oR][oC];
//                                kernelGrad->elem[kD][kR][kC]+=outputDelta->elem[oR][oC]*input->elem[kD][iR+kR][iC+kC];
//                        }
//        }
//}
//
//void DroppedBackwardConvolute3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<bool> > &activeOutput){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10 || !activeOutput[oR][oC]) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kD=0; kD<kernel->depth; kD++)
//                    for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                        for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++){
//                                if (!activeInput[kD][iR+kR][iC+kC]) continue;
//                                inputDelta->elem[kD][iR+kR][iC+kC]+=kernel->elem[kD][kR][kC]*outputDelta->elem[oR][oC];
//                                kernelGrad->elem[kD][kR][kC]+=outputDelta->elem[oR][oC]*input->elem[kD][iR+kR][iC+kC];
//                        }
//        }
//}
//
//
//void BackwardConvoluteGrad3D2D(tensor* input, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kD=0; kD<kernel->depth; kD++)
//                    for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                        for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                                kernelGrad->elem[kD][kR][kC]+=outputDelta->elem[oR][oC]*input->elem[kD][iR+kR][iC+kC];
//        }
//}
//
//void DroppedBackwardConvoluteGrad3D2D(tensor* input, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           double* bias, double* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<bool> > &activeOutput){
//    for(int iR=-paddingR, oR=0; iR<input->rows+paddingR-kernel->rows+1; iR+=strideR, ++oR)
//        for(int iC=-paddingC, oC=0; iC<input->cols+paddingC-kernel->cols+1; iC+=strideC, ++oC){
//                if (fabs(outputDelta->elem[oR][oC])<1E-10 || !activeOutput[oR][oC]) continue;
//                *biasGrad+=outputDelta->elem[oR][oC];
//                for(int kD=0; kD<kernel->depth; kD++)
//                    for(int kR=max(0, -iR); kR<min(kernel->rows, input->rows-iR); kR++)
//                        for(int kC=max(0, -iC); kC<min(kernel->cols, input->cols-iC); kC++)
//                            if (activeInput[kD][iR+kR][iC+kC])
//                                kernelGrad->elem[kD][kR][kC]+=outputDelta->elem[oR][oC]*input->elem[kD][iR+kR][iC+kC];
//        }
//}
//
//
//
//void Convolute3D3D(tensor* input, tensor* output, tensor4D* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC){
//    for(int n=0; n<kernel->number; n++)
//        {
//            Convolute3D2D(input,&(output->elem[n]),&(kernel->elem[n]),&(bias->elem[n]),paddingR,paddingC,strideR,strideC);
//        }
//}
//
//void DroppedConvolute3D3D(tensor* input, tensor* output, tensor4D* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput){
//    for(int n=0; n<kernel->number; n++)
//        {
//            DroppedConvolute3D2D(input,&(output->elem[n]),&(kernel->elem[n]),&(bias->elem[n]),paddingR,paddingC,strideR,strideC, activeInput, activeOutput[n]);
//        }
//}
//
//void BackwardConvolute3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//    for(int n=0; n<kernel->number; n++){
//        BackwardConvolute3D2D(input, inputDelta, &(outputDelta->elem[n]), &(kernel->elem[n]), &(kernelGrad->elem[n]), &(bias->elem[n]),
//                                &(biasGrad->elem[n]), paddingR, paddingC, strideR, strideC);
//    }
//}
//
//void DroppedBackwardConvolute3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput){
//    for(int n=0; n<kernel->number; n++){
//        DroppedBackwardConvolute3D2D(input, inputDelta, &(outputDelta->elem[n]), &(kernel->elem[n]), &(kernelGrad->elem[n]), &(bias->elem[n]),
//                                &(biasGrad->elem[n]), paddingR, paddingC, strideR, strideC, activeInput, activeOutput[n]);
//    }
//}
//
//void BackwardConvoluteGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC){
//    for(int n=0; n<kernel->number; n++){
//        BackwardConvoluteGrad3D2D(input, &(outputDelta->elem[n]), &(kernel->elem[n]), &(kernelGrad->elem[n]), &(bias->elem[n]),
//                                &(biasGrad->elem[n]), paddingR, paddingC, strideR, strideC);
//    }
//}
//
//void DroppedBackwardConvoluteGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput){
//    for(int n=0; n<kernel->number; n++){
//        DroppedBackwardConvoluteGrad3D2D(input, &(outputDelta->elem[n]), &(kernel->elem[n]), &(kernelGrad->elem[n]), &(bias->elem[n]),
//                                &(biasGrad->elem[n]), paddingR, paddingC, strideR, strideC, activeInput, activeOutput[n]);
//    }
//}
//
//
//
//void MaxPool2D(matrix* input, matrix* output, int ** rowInd, int **colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC){
//    double maxVal;
//    int maxR, maxC;
//
//    for(int iR=0, oR=0; iR<input->rows-kernelRsize+1; iR+=strideR, ++oR)
//        for(int iC=0, oC=0; iC<input->cols-kernelCsize+1; iC+=strideC, ++oC){
//                maxVal=input->elem[iR][iC];
//                maxR=iR; maxC=iC;
//                for(int kR=0; kR<kernelRsize; kR++)
//                    for(int kC=0; kC<kernelCsize; kC++)
//                        if (input->elem[iR+kR][iC+kC]>maxVal){
//                            maxVal=input->elem[iR+kR][iC+kC];
//                            maxR=iR+kR;
//                            maxC=iC+kC;
//                        }
//
//                output->elem[oR][oC]+=maxVal;
//                rowInd[oR][oC]=maxR;
//                colInd[oR][oC]=maxC;
//        }
//}
//
//void DroppedMaxPool2D(matrix* input, matrix* output, int ** rowInd, int **colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC,
//             vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput){
//    double maxVal;
//    int maxR, maxC;
//    bool activated;
//
//    for(int iR=0, oR=0; iR<input->rows-kernelRsize+1; iR+=strideR, ++oR)
//        for(int iC=0, oC=0; iC<input->cols-kernelCsize+1; iC+=strideC, ++oC){
//                if (!activeOutput[oR][oC]) continue;
//
//                activated = 0;
//                maxVal=-1E20;
//
//
//                for(int kR=0; kR<kernelRsize; kR++)
//                    for(int kC=0; kC<kernelCsize; kC++)
//                        if (activeInput[iR+kR][iC+kC] && input->elem[iR+kR][iC+kC]>maxVal){
//                            activated=1;
//                            maxVal=input->elem[iR+kR][iC+kC];
//                            maxR=iR+kR;
//                            maxC=iC+kC;
//                        }
//                if (activated){
//                    output->elem[oR][oC]+=maxVal;
//                    rowInd[oR][oC]=maxR;
//                    colInd[oR][oC]=maxC;
//                }
//                else{
//                    activeOutput[oR][oC]=0;
//                }
//        }
//}
//
//
//void BackwardMaxPool2D(matrix* inputDelta, matrix* outputDelta, int** rowInd, int**colInd){
//    for(int r=0; r<outputDelta->rows; r++)
//        for(int c=0; c<outputDelta->cols; c++)
//            inputDelta->elem[rowInd[r][c] ] [colInd[r][c] ] += outputDelta->elem[r][c];
//}
//
//void DroppedBackwardMaxPool2D(matrix* inputDelta, matrix* outputDelta, int** rowInd, int**colInd,
//                              vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput){
//    for(int r=0; r<outputDelta->rows; r++)
//        for(int c=0; c<outputDelta->cols; c++)
//            if (activeOutput[r][c])
//                inputDelta->elem[rowInd[r][c] ] [colInd[r][c] ] += outputDelta->elem[r][c];
//}
//
//
//
//void MaxPool3D(tensor* input, tensor* output, int *** rowInd, int *** colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC){
//    for(int d=0; d<input->depth; d++)
//        MaxPool2D(&(input->elem[d]), &(output->elem[d]), rowInd[d], colInd[d], kernelRsize, kernelCsize, strideR, strideC);
//}
//
//void DroppedMaxPool3D(tensor* input, tensor* output, int *** rowInd, int *** colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC,
//             vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput){
//    for(int d=0; d<input->depth; d++)
//        DroppedMaxPool2D(&(input->elem[d]), &(output->elem[d]), rowInd[d], colInd[d],
//                         kernelRsize, kernelCsize, strideR, strideC, activeInput[d], activeOutput[d]);
//}
//
//void BackwardMaxPool3D(tensor* inputDelta, tensor* outputDelta, int *** rowInd, int *** colInd){
//    for(int d=0; d<inputDelta->depth; d++)
//        BackwardMaxPool2D(&(inputDelta->elem[d]), &(outputDelta->elem[d]), rowInd[d], colInd[d]);
//}
//
//void DroppedBackwardMaxPool3D(tensor* inputDelta, tensor* outputDelta, int *** rowInd, int *** colInd,
//                              vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput){
//    for(int d=0; d<inputDelta->depth; d++)
//        DroppedBackwardMaxPool2D(&(inputDelta->elem[d]), &(outputDelta->elem[d]), rowInd[d], colInd[d], activeInput[d], activeOutput[d]);
//}

double sqr(double x){
    return x*x;
}

int sign(double x){
    if (x>0) return 1;
    return -1;
}

int power(int base, int degree){
    int res = 1;
    for(int d=0; d<degree; ++d)
        res *= base;
    return res;
}
