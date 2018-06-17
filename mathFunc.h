#ifndef __math_func__
#define __math_func__
#include <vector>
using namespace std;

class realNumber;
class vect;
class matrix;
class tensor;
class tensor4D;
class activityMatrix;
class activityTensor;
class orderedData;
class activityData;


struct mathFunc{
    virtual float   f(const float &x)=0;
    virtual float  df(const float &x)=0;
    virtual float ddf(const float &x)=0;
};

struct Sigma: public mathFunc{
    float   f(const float &x);
    float  df(const float &x);
    float ddf(const float &x);
};

struct Tanh: public mathFunc{
    float   f(const float &x);
    float  df(const float &x);
    float ddf(const float &x);
};

struct Relu: public mathFunc{
    float   f(const float &x);
    float  df(const float &x);
    float ddf(const float &x);
};



void SoftMax(orderedData* inp, orderedData* out);




void Convolute2D2D(matrix* input, matrix* output, matrix* reversedKernel, realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvolute2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);


void BackwardConvoluteGrad2D2D(matrix* input, matrix* outputDelta, matrix* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow);




void ConvoluteQuadratic2D2D(matrix* input, matrix* output, matrix* linearReversedKernel, matrix* quadraticReversedKernel,
                            realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvoluteQuadratic2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* linearKernel, matrix* quadraticKernel,
                                    matrix* linearKernelGrad, matrix* quadraticKernelGrad, realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);

void BackwardConvoluteQuadraticGrad2D2D(matrix* input, matrix* outputDelta, matrix* linearKernelGrad, matrix* quadraticKernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow);




void Convolute2D3D(matrix* input, tensor* output, tensor* reversedKernel, vect* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvolute2D3D(matrix* input, matrix* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);

void BackwardConvoluteGrad2D3D(matrix* input, tensor* outputDelta, tensor* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);



void Convolute3D2D(tensor* input, matrix* output, tensor* reversedKernel, realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvolute3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);

void BackwardConvoluteGrad3D2D(tensor* input, matrix* outputDelta, tensor* kernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow);


void ConvoluteStandard3D2D_2_2(tensor* input, matrix* output, tensor* kernel, realNumber* bias);

void BackwardConvoluteStandard3D2D_2_2(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, realNumber* biasGrad);

void BackwardConvoluteGradStandard3D2D_2_2(tensor* input, matrix* outputDelta, tensor* kernelGrad, realNumber* biasGrad);






void ConvoluteQuadratic3D2D(tensor* input, matrix* output, tensor* linearReversedKernel, tensor* quadraticReversedKernel,
                             realNumber* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvoluteQuadratic3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* linearKernel, tensor* quadraticKernel,
                                    tensor* linearKernelGrad, tensor* quadraticKernelGrad, realNumber* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);

void BackwardConvoluteQuadraticGrad3D2D(tensor* input, matrix* outputDelta, tensor* linearKernelGrad, tensor* quadraticKernelGrad,
                           realNumber* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow);




void Convolute3D3D(tensor* input, tensor* output, tensor4D* reversedKernel, vect* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvolute3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);

void BackwardConvoluteGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* kernelGrad,
                           vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);



void ConvoluteQuadratic3D3D(tensor* input, tensor* output, tensor4D* linearReversedKernel, tensor4D* quadraticReversedKernel,
                            vect* bias, int paddingR, int paddingC, vector<int> & indexInputRow);

void BackwardConvoluteQuadratic3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* linearKernel, tensor4D* quadraticKernel,
                                       tensor4D* linearKernelGrad, tensor4D* quadraticKernelGrad, vect* biasGrad, int paddingR, int paddingC, vector<int> &indexOutputRow);

void BackwardConvoluteQuadraticGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* linearKernelGrad, tensor4D* quadraticKernelGrad,
                                        vect* biasGrad, int paddingR, int paddingC, vector<int> & indexOutputRow);


void SequentialMaxMinStandard(tensor* input, vect* kernel, vect* bias, int startDepth, int nConvolutions, activityData* inputActivity, bool testMode);

void BackwardSequentialMaxMinStandard(tensor* input, tensor* inputDelta, vect* kernel, vect* kernelGrad, vect* biasGrad, int startDepth, int nConvolutions, activityData* inputActivity);

void SequentialConvolution(tensor* input, tensor* reversedKernel, vect* bias, int paddingR, int paddingC, vector<int> &indexInputRow, int startDepth, activityData* inputActivity);
void BackwardSequentialConvolution(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                                   int paddingR, int paddingC, vector<int> & indexInputRow, int startDepth, bool computeLastDelta, activityData* inputActivity);

void SequentialConvolutionStandard(tensor* input, tensor* kernel, vect* bias, int startDepth, activityData* inputActivity);
void BackwardSequentialConvolutionStandard(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                                   int startDepth, bool computeLastDelta, activityData* inputActivity);


void SequentialConvolutionMultipleStandard(tensor* input, tensor* kernel, vect* bias, int startDepth, int nConvolutions, activityData* inputActivity);
void BackwardSequentialConvolutionMultipleStandard(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                                   int startDepth, int nConvolutions, bool computeLastDelta, activityData* inputActivity);

void SequentialBottleneckConvolutionStandard(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                                             tensor* verticalConv, int startDepth, int nConvolutions, activityData* inputActivity, bool testMode);

void SequentialBottleneckConvolutionStandardLimited(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                                             tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, bool testMode);

void SequentialBottleneckConvolutionStandardRandom(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices, bool testMode);

void SequentialBottleneckConvolutionStandardRandomSymmetric(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity,
                activityData* verticalConvActivity, int* indices, bool testMode);


void SequentialBottleneckConvolutionStandardRandomSymmetricNoBias(tensor* input, tensor* kernel_vert, tensor* kernel_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity,
                activityData* verticalConvActivity, int* indices, bool testMode);



void SequentialBottleneckConvolutionStandardRandomFullySymmetric(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices, bool testMode);

void SequentialBottleneckConvolutionStandardRandomLimited(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                tensor* verticalConv, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int* indices, bool testMode);



void BackwardSequentialBottleneckConvolutionStandard(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                                                   tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                                                   tensor* vertConvolutionGrad, int startDepth, int nConvolutions, activityData* inputActivity);

void BackwardSequentialBottleneckConvolutionStandardLimited(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                            tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                            tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity);


void BackwardSequentialBottleneckConvolutionStandardRandom(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                    tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                    tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int * indices);

void BackwardSequentialBottleneckConvolutionStandardRandomSymmetric(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                    tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution, tensor* vertConvolutionGrad, int startDepth,
                    int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, activityData* verticalConvActivity, int* indices);


void BackwardSequentialBottleneckConvolutionStandardRandomSymmetricNoBias(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                    tensor* kernel_hor, tensor* kernelGrad_hor, tensor* vertConvolution, tensor* vertConvolutionGrad, int startDepth,
                    int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, activityData* verticalConvActivity, int* indices);


void BackwardSequentialBottleneckConvolutionStandardRandomFullySymmetric(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                    tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                    tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int * indices);

void BackwardSequentialBottleneckConvolutionStandardRandomLimited(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                    tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                    tensor* vertConvolutionGrad, int startDepth, int nConvolutions, int limitDepth, int alwaysPresentDepth, activityData* inputActivity, int * indices);

void BackwardSequentialBottleneckConvolutionStandardPartialGrad(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                                                   tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                                                   tensor* vertConvolutionGrad, int startDepth, int nConvolutions, activityData* inputActivity);


void SequentialBottleneckConvolutionReluStandard(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias_hor,
                                             tensor* verticalConv, int startDepth, int nConvolutions, activityData* inputActivity);


void BackwardSequentialBottleneckConvolutionReluStandard(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                                                   tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad_hor, tensor* vertConvolution,
                                                   tensor* vertConvolutionGrad, int startDepth, int nConvolutions, activityData* inputActivity);

void BackwardVertical2D(matrix* input, matrix* inputDelta, matrix* outputDelta, float kernel, float & kernelGrad);

void ConvolutionStandardVertical1D(tensor* input, tensor* kernel, matrix* output);
void ConvolutionStandardVerticalRandom1D(tensor* input, int* indices, tensor* kernel, matrix* output);

void AddConvolutionStandardVertical1D(tensor* input, tensor* kernel, matrix* output);
void BackwardConvolutionStandardVertical1D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad);
void BackwardConvolutionStandardVerticalRandom1D(tensor* input, int* indices, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad);


void BackwardConvolutionStandardVertical1DGrad(tensor* input, matrix* outputDelta, tensor* kernelGrad);
void BackwardConvolutionStandardVertical1DPartialGrad(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, int startDepth);

void BottleneckConvolutionStandard(tensor* input, matrix* vertOutput, matrix* convOutput, tensor* kernel_vert, matrix* kernel_hor, realNumber* bias_hor);

void BottleneckConvolutionStandardRandom(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, realNumber* biasHor);

void BottleneckConvolutionStandardRandomSymmetric(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, realNumber* biasHor);

void BottleneckConvolutionStandardRandomSymmetricDrop(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert,
                                                  matrix* kernelHor, realNumber* biasHor, activityData* vertOutputActivity);

void BottleneckConvolutionStandardRandomSymmetricNoBiasDrop(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert,
                                                  matrix* kernelHor, activityData* vertOutputActivity);


void SymmetricConvolution(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, int symmetryLevel);
void BackwardSymmetricConvolution(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput,
                                                    matrix* vertOutputDelta, tensor* kernelVert, matrix* kernelHor,
                                                    tensor* kernelGradVert, matrix* kernelGradHor, int symmetryLevel);



void SymmetricConvolution3D(tensor* input, tensor* output, tensor* kernel, vect* bias, int symmetryLevel);
void BackwardSymmetricConvolution3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad, int symmetryLevel);


void VerticalConvolution3D(tensor* input, tensor* output, tensor* kernel);
void BackwardVerticalConvolution3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad);




void BottleneckConvolutionStandardRandomFullySymmetric(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, realNumber* biasHor);

void BottleneckConvolutionStandardLimited(tensor* inputStart, tensor* inputLast, matrix* vertOutput, matrix* convOutput, tensor* kernelVertStart,
                                          tensor* kernelVertLast, matrix* kernelHor, realNumber* biasHor);



void BackwardBottleneckConvolutionStandard(tensor* input, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernel_vert, matrix* kernel_hor, tensor* kernelGrad_vert, matrix* kernelGrad_hor, realNumber* biasGrad_hor);

void BackwardBottleneckConvolutionStandardRandom(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                                    tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, realNumber* biasGradHor);

void BackwardBottleneckConvolutionStandardRandomSymmetric(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                                    tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, realNumber* biasGradHor);

void BackwardBottleneckConvolutionStandardRandomSymmetricDrop(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput,
                                                    matrix* vertOutputDelta, tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor,
                                                    realNumber* biasGradHor, activityData* vertOutputActivity);

void BackwardBottleneckConvolutionStandardRandomSymmetricNoBiasDrop(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput,
                                                    matrix* vertOutputDelta, tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor,
                                                    activityData* vertOutputActivity);

void BackwardBottleneckConvolutionStandardRandomFullySymmetric(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                                    tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, realNumber* biasGradHor);

void BackwardBottleneckConvolutionStandardLimited(tensor* inputStart, tensor* inputLast, tensor* inputLastDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernelVertLast, matrix* kernel_hor, tensor* kernelGradVertStart, tensor* kernelGradVertLast, matrix* kernelGradHor, realNumber* biasGradHor);

void BackwardBottleneckConvolutionStandardLimitedFull(tensor* inputStart, tensor* inputLast, tensor* inputStartDelta, tensor* inputLastDelta,
                            matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta, tensor* kernelVertStart,
                            tensor* kernelVertLast, matrix* kernelHor, tensor* kernelGradVertStart, tensor* kernelGradVertLast, matrix* kernelGradHor, realNumber* biasGradHor);

void BackwardBottleneckConvolutionStandardGrad(tensor* input, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernel_vert, matrix* kernel_hor, tensor* kernelGrad_vert, matrix* kernelGrad_hor, realNumber* biasGrad_hor);

void BackwardBottleneckConvolutionStandardPartialGrad(tensor* input, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                            tensor* kernel_vert, matrix* kernel_hor, tensor* kernelGrad_vert, matrix* kernelGrad_hor, realNumber* biasGrad_hor, int startDepth);

//void ForwardConvoluteStandardByIndex(float* input, float* output, const int siz, float* revKernel, const vector<int> & indexInputRow);

void ConvoluteStandard(float* input, float* output, const int siz, float* kernel);
void ConvoluteStandardSymmetric(float* input, float* output, const int siz, float* kernel);
void ConvoluteStandardFullySymmetric0(float* input, float* output, const int siz, float* kernel);


void ConvoluteStandard2D2D(matrix* input, matrix* output, matrix* kernel);
void ConvoluteStandard2D2DSymmetric(matrix* input, matrix* output, matrix* kernel);
void ConvoluteStandard2D2DSymmetric2(matrix* input, matrix* output, matrix* kernel);
void ConvoluteStandard2D2DSymmetric3(matrix* input, matrix* output, matrix* kernel);
void ConvoluteStandard2D2DFullySymmetric(matrix* input, matrix* output, matrix* kernel);

//void ForwardConvoluteStandardCreateIndex(float* input, float* output, const int siz, float* revKernel, vector<int> & indexInputRow);

void BackwardConvoluteStandard(float* input, float* inputDelta, float* outputDelta, float* kernel, float* kernelGrad, const int siz);
void BackwardConvoluteStandardSymmetric(float* input, float* inputDelta, float* outputDelta, float* kernel, float* kernelGrad, const int siz);
void BackwardConvoluteStandardFullySymmetric0(float* input, float* inputDelta, float* outputDelta, float* kernel, float* kernelGrad, const int siz);

void BackwardConvoluteGradStandard(float* input, float* outputDelta, float* kernelGrad, const int siz);
void BackwardConvoluteStandard2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteStandard2D2DSymmetric(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteStandard2D2DSymmetric2(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteStandard2D2DSymmetric3(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteStandard2D2DFullySymmetric(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteGradStandard2D2D(matrix* input, matrix* outputDelta, matrix* kernelGrad, realNumber* biasGrad);

void BackwardConvoluteStandard3D2D_4_4(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteGradStandard3D2D_4_4(tensor* input, matrix* outputDelta, tensor* kernelGrad, realNumber* biasGrad);

void BackwardConvoluteStandard2D2D_4_4(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, realNumber* biasGrad);

void BackwardConvoluteStandard_4(float* input, float* inputDelta, float* outputDelta, float* kernel, float* kernelGrad);
void BackwardConvoluteGradStandard_4(float* input, float* outputDelta, float* kernelGrad);
//void ConvoluteStandard2D2D(matrix* input, matrix* output, matrix* reversedKernel, realNumber* bias, vector<int> & indexInputRow);

void ConvoluteStandard3D2D(tensor* input, matrix* output, tensor* kernel, realNumber* bias);

void ConvoluteMultipleStandard3D3D(tensor* input, tensor* output, tensor* kernel, vect* bias);
void BackwardConvoluteMultipleStandard3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad);


void BackwardConvoluteStandard3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, realNumber* biasGrad);
void BackwardConvoluteGradStandard3D2D(tensor* input, matrix* outputDelta, tensor* kernelGrad, realNumber* biasGrad);

void AveragePool2D(matrix* input, matrix* output, int kernelR, int kernelC);

void BackwardAveragePool2D(matrix* inputDelta, matrix* outputDelta, int kernelR, int kernelC);


void AveragePool3D(tensor* input, tensor* output, int kernelR, int kernelC);

void BackwardAveragePool3D(tensor* inputDelta, tensor* outputDelta, int kernelR, int kernelC);


void AveragePool3D_2_2(tensor* input, tensor* output);
void AveragePool3D_2_2(tensor* input, tensor* output, int numNonZero);

void BackwardAveragePool3D_2_2(tensor* inputDelta, tensor* outputDelta);
void BackwardAveragePool3D_2_2(tensor* inputDelta, tensor* outputDelta, int numNonZero);


void BackwardAveragePool2D_2_2(matrix* inputDelta, matrix* outputDelta);
void BackwardAveragePool2D_2_2(matrix* inputDelta, matrix* outputDelta, int numNonZero);

void AveragePool2D_2_2(matrix* input, matrix* output);
void AveragePool2D_2_2(matrix* input, matrix* output, int numNonZero);



void MaxPool2D(matrix* input, matrix* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMaxPool2D(matrix* inputDelta, matrix* outputDelta, int* rowInd, int*colInd);

void MaxAbsPool2D(matrix* input, matrix* output, matrix* maxAbs, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMaxAbsPool2D(matrix* inputDelta, matrix* outputDelta, int* rowInd, int*colInd);

void MaxAbsPool2D_2_2(matrix* input, matrix* output, matrix* maxAbs, int* rowInd, int *colInd);

//void MaxPoolSoftMaxIndex2D(matrix* input, float& output, float& softMaxRow, float& softMaxCol,
//                           matrix* softMaxInput, int& rowInd, int& colInd, int kernelRsize, int kernelCsize);

void MaxPool3D(tensor* input, tensor* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMaxPool3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd);

void MaxAbsPool3D(tensor* input, tensor* output, tensor* maxAbs, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMaxAbsPool3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd);

void MaxAbsPool3D_2_2(tensor* input, tensor* output, tensor* maxAbs, int* rowInd, int *colInd);

void MaxAbsPoolIndex3D(tensor* input, tensor* output, tensor* maxAbs, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMaxAbsPoolIndex3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd);

void MinPool2D(matrix* input, matrix* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMinPool2D(matrix* inputDelta, matrix* outputDelta, int* rowInd, int*colInd);

void MinPool3D(tensor* input, tensor* output, int* rowInd, int *colInd, int kernelRsize, int kernelCsize);
void BackwardMinPool3D(tensor* inputDelta, tensor* outputDelta, int* rowInd, int* colInd);

void CalculateSoftIndex(matrix* input, matrix* softMaxInput, float & softMaxRow, float & softMaxCol, float & maxVal, float logFactor);
void MaxAbsPoolSoftIndex3D(tensor* input, vect* output, vect* maxAbs, tensor* softMaxInput, int* rowInd, int* colInd, float logFactor);
void BackwardSoftIndex(matrix* input, matrix* softMaxInput, const float softRow, const float softCol, matrix* inputDelta,
                       const float deltaSoftRow, const float deltaSoftCol, float logFactor);
void BackwardMaxAbsPoolSoftIndex3D(tensor* input, tensor* softMaxInput, vect* output, tensor* inputDelta, vect* outputDelta,
                                   int* rowInd, int* colInd, float logFactor);

void MaxAbsPoolSoftDiffIndex3D(tensor* input, vect* output, vect* tempOutput, vect* maxAbs, tensor* softMaxInput, int* rowInd, int* colInd, float logFactor);
void BackwardMaxAbsPoolSoftDiffIndex3D(tensor* input, tensor* softMaxInput, vect* tempOutput, tensor* inputDelta, vect* outputDelta, vect* tempOutputDelta,
                                   int* rowInd, int* colInd, float logFactor);


void MeanVarPool(orderedData* input, float * mean, float * var);
void MeanVarPoolTensor(tensor* input, tensor* output);
void BackwardMeanVarPool(orderedData* input, float * mean, float * var, orderedData* inputDelta, float * meanDelta, float * varDelta);
void BackwardMeanVarPoolTensor(tensor* input, tensor* output, tensor* inputDelta, tensor* outputDelta);

void MeanQuadStatsPool(matrix* input, float * stats);
void MeanQuadStatsPoolTensor(tensor* input, tensor* output);
void BackwardMeanQuadStatsPool(matrix* input, float * stats, matrix* inputDelta, float * statsDelta);
void BackwardMeanQuadStatsPoolTensor(tensor* input, tensor* output, tensor* inputDelta, tensor* outputDelta);



void MeanStDevPool(orderedData* input, float * mean, float * stDev);
void MeanStDevPoolTensor(tensor* input, tensor* output);
void BackwardMeanStDevPool(orderedData* input, float * mean, float * stDev, orderedData* inputDelta, float * meanDelta, float * stDevDelta);
void BackwardMeanStDevPoolTensor(tensor* input, tensor* output, tensor* inputDelta, tensor* outputDelta);


void AverageMaxAbsPool(tensor* input, tensor* output, int kernelRsize, int kernelCsize, int * rowInd, int * colInd, int onlyAveragePoolingDepth);
void AverageMaxAbsPoolMatrixAll(matrix* input, float * outAverage, float * outMaxbs, int * rowInd, int * colInd);
void AverageMaxAbsPoolMatrix_2_2(matrix* input, matrix* outputAverage, matrix* outputMaxAbs, int * rowInd, int * colInd);

void BackwardAverageMaxAbsPool(tensor* inputDelta, tensor* outputDelta,
                               int kernelRsize, int kernelCsize, int * rowInd, int * colInd, int onlyAveragePoolingDepth);


void CenterPool(matrix* input, float * centers, float * sum);
void CenterPoolTensor(tensor* input, tensor* output, vect* sum);
void BackwardCenterPool(matrix * input, float * centers, float * sum, matrix * inputDelta, float * centersDelta);
void BackwardCenterPoolTensor(tensor * output, vect * sum, tensor * inputDelta, tensor * outputDelta);


void BackwardMedianPoolTensor(tensor * inputDelta, tensor * outputDelta, int * index);
void BackwardMedianNonzeroPoolTensor(tensor* output, tensor * inputDelta, tensor * outputDelta, int * index);

void MedianNonzeroPoolTensor(tensor * input, tensor * output, int * index);
void MedianPoolTensor(tensor * input, tensor * output, int * index);

void QuartilesPoolTensor(tensor * input, tensor * output, int * index, int numQuartiles);
void QuartilesNonzeroPoolTensor(tensor * input, tensor * output, int * index, int numQuartiles);
void BackwardQuartilesPoolTensor(tensor * inputDelta, tensor * outputDelta, int * index, int numQuartiles);
void BackwardQuartilesNonzeroPoolTensor(tensor * output, tensor * inputDelta, tensor * outputDelta, int * index, int numQuartiles);

void ForwardStairsConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, activityData* verticalConvActivity, int* indices, bool testMode);

void BackwardStairsConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, activityData* verticalConvActivity, int* indices);



void ForwardStairsFullBottleneck(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias, tensor* verticalConv,
                                int startDepth, int numStairs, int numStairConvolutions, int bottleneckDepth,
                                activityData* inputActivity, bool testMode);

void BackwardStairsFullBottleneck(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                                  tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad,
                                tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                                int numStairConvolutions, int bottleneckDepth, activityData* inputActivity);



void ForwardStairsFullBottleneckBalancedDrop(tensor* input, tensor* kernel_vert, tensor* kernel_hor, vect* bias, tensor* verticalConv,
                                int startDepth, int numStairs, int numStairConvolutions, int bottleneckDepth,
                                activityData* inputActivity, tensor* multipliers, bool testMode);

void BackwardStairsFullBottleneckBalancedDrop(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert,
                                  tensor* kernel_hor, tensor* kernelGrad_hor, vect* biasGrad,
                                tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                                int numStairConvolutions, int bottleneckDepth, activityData* inputActivity, tensor* multipliers);





void ForwardStairsSymmetricConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, int* indices, bool testMode, int symmetryLevel);

void BackwardStairsSymmetricConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, int* indices, int symmetryLevel);





void ForwardStairsFullConvolution(tensor* input, tensor* kernel, vect* bias, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, bool testMode, int symmetryLevel);

void BackwardStairsFullConvolution(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, int symmetryLevel);






void ForwardStairsFullConvolutionBalancedDrop(tensor* input, tensor* kernel, vect* bias, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, tensor* multipliers, bool testMode, int symmetryLevel);

void BackwardStairsFullConvolutionBalancedDrop(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, tensor* multipliers, int symmetryLevel);




void ForwardStairsFullConvolutionRelu(tensor* input, tensor* kernel, vect* bias, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, bool testMode, int symmetryLevel);

void BackwardStairsFullConvolutionRelu(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad, vect* biasGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, int symmetryLevel);




void ForwardStairsSymmetricConvolutionRelu(tensor* input, tensor* kernel, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, bool testMode, int symmetryLevel);

void BackwardStairsSymmetricConvolutionRelu(tensor* input, tensor* inputDelta, tensor* kernel, tensor* kernelGrad,
                               int startDepth, int numStairs, int numStairConvolutions, activityData* inputActivity, int symmetryLevel);






void ForwardStairsSequentialConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, int* indices, bool testMode, int symmetryLevel);

void BackwardStairsSequentialConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, int* indices, int symmetryLevel);




void ForwardStairsPyramidalConvolution(tensor* input, tensor* kernel_vert, tensor* kernel_hor, tensor* verticalConv, int startDepth, int numStairs, int numStairConvolutions,
                             activityData* inputActivity, int* indices, bool testMode);

void BackwardStairsPyramidalConvolution(tensor* input, tensor* inputDelta, tensor* kernel_vert, tensor* kernelGrad_vert, tensor* kernel_hor, tensor* kernelGrad_hor,
                               tensor* verticalConv, tensor* verticalConvDelta, int startDepth, int numStairs,
                               int numStairConvolutions, activityData* inputActivity, int* indices);

void PyramidalConvolution(tensor* input, int* indices, matrix* vertOutput, matrix* convOutput, tensor* kernelVert, matrix* kernelHor, int stair);
void PyramidalVerticalConvolution(tensor* input, int* indices, tensor* kernel, matrix* output, int stair);
void PyramidalConvolution2D2D(matrix* input, matrix* output, matrix* kernel, int stair);
void ConvoluteSymmetricNoPadding(float* input, float* output, const int siz, float* kernel);


void BackwardPyramidalConvolution(tensor* input, int* indices, tensor* inputDelta, matrix* convOutputDelta, matrix* vertOutput, matrix* vertOutputDelta,
                                  tensor* kernelVert, matrix* kernelHor, tensor* kernelGradVert, matrix* kernelGradHor, int stair);
void BackwardPyramidalConvolution2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad, int stair);
void BackwardConvoluteSymmetricNoPadding(float* input, float* inputDelta, float* outputDelta, float* kernel, float* kernelGrad, const int siz);
void BackwardPyramidalVerticalConvolution(tensor* input, int* indices, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad, int stair);

void FullSubAveragePool(tensor* input, tensor* output, int border);
void FullSubAveragePool2D(matrix* input, float* output, int border);
void BackwardFullSubAveragePool2D(matrix* inputDelta, float * outputDelta, int border);
void BackwardFullSubAveragePool(tensor* inputDelta, tensor* outputDelta, int border);

void AveragePool3D_all(tensor* input, tensor* output);
void BackwardAveragePool3D_all(tensor* inputDelta, tensor* outputDelta);

void AveragePool3D_all(tensor* input, tensor* output, int numNonZero);
void BackwardAveragePool3D_all(tensor* inputDelta, tensor* outputDelta, int numNonZero);

void DeleteOnlyShell(orderedData* link);
void DeleteOnlyShellActivity(activityData* link);

float sqr(float x);
int sign(float x);
int power(int base, int degree);

float TrustRegionFunc(float lamb, vect* eigenValues, vect* rV, float eps);
float TrustRegionFuncDeriv(float lamb, vect* eigenValues, vect* rV, float eps);

void FillRandom(int* index, int startInd, int endInd, int len);
void Switch(int & a, int & b);
//
//void Convolute2D2D(matrix* input, matrix* output, matrix* kernel, float* bias, int paddingR, int paddingC, int strideR, int strideC);
//void Convolute2D3D(matrix* input, tensor* output, tensor* kernel, vect*   bias, int paddingR, int paddingC, int strideR, int strideC);
//void Convolute3D2D(tensor* input, matrix* output, tensor* kernel, float* bias, int paddingR, int paddingC, int strideR, int strideC);
//void Convolute3D3D(tensor* input, tensor* output, tensor4D* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC);
//
//
//void BackwardConvolute2D2D(matrix* input,matrix* inputDelta,matrix* outputDelta,matrix* kernel,matrix* kernelGrad,
//                           float* bias, float* biasGrad,int paddingR,int paddingC,int strideR,int strideC);
//
//void BackwardConvolute2D3D(matrix* input, matrix* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC);
//
//void BackwardConvolute3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           float* bias, float* biasGrad, int paddingR, int paddingC, int strideR, int strideC);
//
//void BackwardConvolute3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC);
//
//
//void BackwardConvoluteGrad2D2D(matrix* input, matrix* outputDelta,matrix* kernel,matrix* kernelGrad,
//                           float* bias, float* biasGrad,int paddingR,int paddingC,int strideR,int strideC);
//
//void BackwardConvoluteGrad2D3D(matrix* input, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC);
//
//void BackwardConvoluteGrad3D2D(tensor* input, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           float* bias, float* biasGrad, int paddingR, int paddingC, int strideR, int strideC);
//
//void BackwardConvoluteGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC);
//
//void DroppedConvolute2D2D(matrix* input, matrix* output, matrix* kernel, float* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput);
//
//void DroppedBackwardConvolute2D2D(matrix* input, matrix* inputDelta, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
//                           float* bias, float* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput);
//
//void DroppedBackwardConvoluteGrad2D2D(matrix* input, matrix* outputDelta, matrix* kernel, matrix* kernelGrad,
//                           float* bias, float* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput);
//
//
//void DroppedConvolute2D3D(matrix* input, tensor* output, tensor* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<bool> > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//
//void DroppedBackwardConvolute2D3D(matrix* input, matrix* inputDelta, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//
//void DroppedBackwardConvoluteGrad2D3D(matrix* input, tensor* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<bool> > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//
//
//void DroppedConvolute3D2D(tensor* input, matrix* output, tensor* kernel, float* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<vector<bool> > > &activeInput, vector<vector<bool> > &activeOutput);
//
//void DroppedBackwardConvolute3D2D(tensor* input, tensor* inputDelta, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           float* bias, float* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<bool> > &activeOutput);
//
//void DroppedBackwardConvoluteGrad3D2D(tensor* input, matrix* outputDelta, tensor* kernel, tensor* kernelGrad,
//                           float* bias, float* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<bool> > &activeOutput);
//
//
//void DroppedConvolute3D3D(tensor* input, tensor* output, tensor4D* kernel, vect* bias, int paddingR, int paddingC, int strideR, int strideC,
//                          vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//
//void DroppedBackwardConvolute3D3D(tensor* input, tensor* inputDelta, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//
//void DroppedBackwardConvoluteGrad3D3D(tensor* input, tensor* outputDelta, tensor4D* kernel, tensor4D* kernelGrad,
//                           vect* bias, vect* biasGrad, int paddingR, int paddingC, int strideR, int strideC,
//                           vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//
//
//
//
//void MaxPool2D(matrix* input, matrix* output, int ** rowInd, int **colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC);
//void BackwardMaxPool2D(matrix* inputDelta, matrix* outputDelta, int ** rowInd, int ** colInd);
//
//void DroppedMaxPool2D(matrix* input, matrix* output, int ** rowInd, int **colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC,
//             vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput);
//
//void DroppedBackwardMaxPool2D(matrix* inputDelta, matrix* outputDelta, int** rowInd, int**colInd,
//                              vector<vector<bool> > &activeInput, vector<vector<bool> > &activeOutput);
//
//
//void MaxPool3D(tensor* input, tensor* output, int *** rowInd, int *** colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC);
//void BackwardMaxPool3D(tensor* inputDelta, tensor* outputDelta, int *** rowInd, int *** colInd);
//
//void DroppedMaxPool3D(tensor* input, tensor* output, int *** rowInd, int *** colInd,
//             int kernelRsize, int kernelCsize, int strideR, int strideC,
//             vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput);
//void DroppedBackwardMaxPool3D(tensor* inputDelta, tensor* outputDelta, int *** rowInd, int *** colInd,
//                              vector<vector<vector<bool> > > &activeInput, vector<vector<vector<bool> > > &activeOutput);



#endif // __math_func__
