#include <math.h>
#include "data.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "matrix.h"
#include "orderedData.h"
#include "computationalNode.h"
#include "vect.h"
#include "tensor.h"
#include "randomGenerator.h"
#include "globals.h"
#include "tensor4D.h"
#include "mathFunc.h"
using namespace std;


int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void Data::Initiate(){
    classData = new tensor4D();
}

void Data::AllocateMemory(int C_, int number_of_images_, int depth_, int rows_, int cols_){
    C = C_;
    siz = new int[C];
    for(int c=0; c<C; ++c)
        siz[c] = 0;
    classData = new tensor4D(number_of_images_, depth_, rows_, cols_);
    labels = new int[number_of_images_];
}


//
//void Data::PreAllocateMemory(int C_, int totalSize_){
//    C = C_;
//    totalSize = totalSize_;
//    siz = new int[C];
//    classData = new orderedData[totalSize];
//    labels = new int[totalSize];
//    for(int j=0; j<C; ++j)
//        siz[j] = 0;
//}
//
//void Data::AllocateMemory(int C_, int mbSize_, int siz_){
//    C=C_;
//    siz=new int[C];
//    classData = new orderedData** [C];
//    for(int c=0; c<C; c++){
//        classData[c]=new orderedData* [mbSize_];
//        siz[c]=siz_;
//    }
//    totalSize=C*siz_;
//}
//
//void Data::AllocateMemory(int C_, int * siz_){
//    C=C_;
//    totalSize=0;
//    siz=new int[C];
//    classData = new orderedData** [C];
//    for(int c=0; c<C; c++){
//        classData[c]=new orderedData* [siz_[c] ];
//        siz[c]=siz_[c];
//        totalSize+=siz[c];
//    }
//}



void Data::ReadTrainingMnist()
{
    char laptop_image_file [] = "/home/slavko/Dropbox/datasets/MNIST/train-images.idx3-ubyte";
    char laptop_label_file [] = "/home/slavko/Dropbox/datasets/MNIST/train-labels.idx1-ubyte";

    char machine_image_file [] = "/home/viacheslavdudar/Dropbox/datasets/MNIST/train-images.idx3-ubyte";
    char machine_label_file [] = "/home/viacheslavdudar/Dropbox/datasets/MNIST/train-labels.idx1-ubyte";

    ifstream file_image, file_label;
    int lab;

    if (LAPTOP) {
        file_image.open(laptop_image_file, ios::binary);
        file_label.open(laptop_label_file, ios::binary);
    }

    else{
        file_image.open(machine_image_file, ios::binary);
        file_label.open(machine_label_file, ios::binary);
    }

    if (!file_image || !file_label) {cout<<"files not found"<<endl; return;}

    int magic_number=0, n_rows=0, n_cols=0, number_of_images;
    unsigned char temp=0, label;

    file_image.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);

    file_label.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);

    file_image.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);

    file_label.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);

    file_image.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);

    file_image.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);

    cout<<number_of_images<<" "<<n_rows<<" "<<n_cols<<endl;
    this->AllocateMemory(10, number_of_images, 1, n_rows, n_cols);
    cout<<classData->number<<" "<<classData->depth<<" "<<classData->rows<<" "<<classData->cols<<" "<<classData->len<<endl;

    for(int i=0; i<number_of_images; ++i)
    {
        file_label.read((char*)&label,sizeof(label));
        lab = int(label);
        labels[i] = lab;
        //cout<<lab<<endl;
        //classData[i] = new tensor(1, n_rows, n_cols);
        //orderedData* tensLink = classData[lab][siz[lab] ];
        siz[lab]++;



        for(int r=0; r<n_rows; ++r)
            for(int c=0; c<n_cols; ++c)
            {
                file_image.read((char*)&temp, sizeof(temp));
                //cout<<"read";
                classData->At(i, 0, r, c) = float(temp)/255.0 - 0.5;
                //cout<<" set"<<endl;
                //tensLink->elem[r*n_cols+c]=float(temp)/255.0 - 0.5;
            }

        //cout<<labels[i]<<" "<<siz[labels[i] ]<<" "<<i<<endl;
    }


    cout<<"Data is read"<<endl;
    cout<<"sizes: "<<endl;
    for(int i=0; i<C; i++)
        cout<<siz[i]<<endl;

    file_image.close();
    file_label.close();
}

void Data::ReadTestMnist(){

    char laptop_image_file [] = "/home/slavko/Dropbox/datasets/MNIST/t10k-images.idx3-ubyte";
    char laptop_label_file [] = "/home/slavko/Dropbox/datasets/MNIST/t10k-labels.idx1-ubyte";

    char machine_image_file [] = "/home/viacheslavdudar/Dropbox/datasets/MNIST/t10k-images.idx3-ubyte";
    char machine_label_file [] = "/home/viacheslavdudar/Dropbox/datasets/MNIST/t10k-labels.idx1-ubyte";

    ifstream file_image, file_label;

    if (LAPTOP) {
        file_image.open(laptop_image_file, ios::binary);
        file_label.open(laptop_label_file, ios::binary);
    }

    else{
        file_image.open(machine_image_file, ios::binary);
        file_label.open(machine_label_file, ios::binary);
    }

    if (!file_image || !file_label) {cout<<"files not found"<<endl; return;}

    int magic_number=0, n_rows=0, n_cols=0, lab, number_of_images;
    unsigned char temp=0, label;

    file_image.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);

    file_label.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);

    file_image.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);

    file_label.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);

    file_image.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);

    file_image.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);

    AllocateMemory(10, number_of_images, 1, n_rows, n_cols);

    //AllocateMemory(10, 1300, 0);
    //totalSize=number_of_images;

    for(int i=0; i<number_of_images; ++i)
    {
        file_label.read((char*)&label,sizeof(label));
        lab=int(label);
        labels[i] = lab;

        //classData[lab][siz[lab] ]=new tensor(1, n_rows, n_cols);
        //orderedData* tensLink=classData[lab][siz[lab] ];
        siz[lab]++;

        for(int r=0; r<n_rows; ++r)
            for(int c=0; c<n_cols; ++c)
            {
                file_image.read((char*)&temp,sizeof(temp));
                classData->At(i,0,r,c) = float(temp)/255.0 - 0.5;
            }
    }
    file_image.close();
    file_label.close();

    cout<<"Data is read"<<endl;
    cout<<"sizes: "<<endl;
    for(int i=0; i<C; i++)
        cout<<siz[i]<<endl;
}



void Data::AddCifar10Batch(char batchFileName[], int startingIndex){
    ifstream batchFile(batchFileName, ios::binary);
    if (!batchFile) cout<<"File not found"<<endl;

    int number_of_images = 10000;
    int n_rows = 32, n_cols = 32;
    int lab;
    unsigned char tplabel, temp;

    for(int i = 0; i < number_of_images; ++i){
        batchFile.read((char*) &tplabel, sizeof(tplabel));
        lab=int(tplabel);
        labels[startingIndex + i] = lab;
        for(int ch = 0; ch < 3; ++ch)
            for(int r = 0; r < n_rows; ++r)
                for(int c = 0; c < n_cols; ++c){
                    batchFile.read((char*) &temp, sizeof(temp));
                    if (!READ_TO_VECT){
                        classData->At(startingIndex + i, ch, r, c) = float(temp)/255.0;
                        if (INPUT_NEG_POS)
                            classData->At(startingIndex + i, ch, r, c) -= 0.5;
                    }

                    else{
                        classData->At(startingIndex + i, c + 32 * r + 1024 * ch, 1, 1) =  float(temp)/255.0;
                        if (INPUT_NEG_POS)
                            classData->At(startingIndex + i, c + 32 * r + 1024 * ch, 1, 1) -= 0.5;
                    }
                }
        ++siz[lab];
    }

    batchFile.close();

}

void Data::ReadTrainingCifar10(){
    if (!READ_TO_VECT)
        AllocateMemory(10, 50000, 3, 32, 32);
    else
        AllocateMemory(10, 50000, 3072, 1, 1);
    //AllocateMemory(10, 5000, 0);

    char laptopBatchFileName[] = "/home/slavko/Dropbox/datasets/CIFAR_10/data_batch_1.bin";
    char machineBatchFileName[] = "/home/viacheslavdudar/Dropbox/datasets/CIFAR_10/data_batch_1.bin";
    char localBatchFileName[] = "CIFAR_10/data_batch_1.bin";
    for(int i=1; i<=5; i++){
        if (LOCAL_DATA){
            localBatchFileName[20] = '0'+i;
            AddCifar10Batch(localBatchFileName, (i-1) * 10000);
        }
        else{
            if (LAPTOP){
                laptopBatchFileName[50]='0'+i;
                AddCifar10Batch(laptopBatchFileName, (i-1) * 10000);
            }

            else{
                machineBatchFileName[59]='0'+i;
                AddCifar10Batch(machineBatchFileName, (i-1) * 10000);
            }
        }

    }

    //totalSize=50000;
    cout<<"Training Cifar 10 is read"<<endl;
    for(int i=0; i<10; i++){
        cout<<siz[i]<<'\t';
    }
    cout<<endl;
}

void Data::ReadTestCifar10(){
    if (!READ_TO_VECT)
        AllocateMemory(10, 10000, 3, 32, 32);
    else
        AllocateMemory(10, 10000, 3072, 1, 1);

    char laptopBatchFileName[]  = "/home/slavko/Dropbox/datasets/CIFAR_10/test_batch.bin";
    char machineBatchFileName[] = "/home/viacheslavdudar/Dropbox/datasets/CIFAR_10/test_batch.bin";
    char localBatchFileName[] = "CIFAR_10/test_batch.bin";
    if (LOCAL_DATA)
         AddCifar10Batch(localBatchFileName, 0);
    else{
        if (LAPTOP)
            AddCifar10Batch(laptopBatchFileName, 0);
        else
            AddCifar10Batch(machineBatchFileName, 0);
    }


    //totalSize=10000;
    cout<<"Test Cifar 10 is read"<<endl;
    for(int i=0; i<10; i++){
        cout<<siz[i]<<'\t';
    }
    cout<<endl;
}



//void Data::SelectRandomMiniBatch(Data* source, int mbSize){
//    int ind;
//    //cout<<C<<" "<<mbSize<<endl;
//    for(int c=0; c<C; c++){
//        siz[c] = mbSize;
//        for(int j=0; j<mbSize; j++){
//            ind=randomGenerator::rand()%source->siz[c];
//            classData[c][j]=source->classData[c][ind];
//        }
//    }
//    totalSize = C * mbSize;
//}

void Data::SelectMiniBatch(Data* source, int startingIndex, int mbSize){
    C = source->C;
    if (startingIndex>source->classData->number)
         cout<<"ERROR: SUBDATA"<<endl;

    classData->Sub4DTensor(source->classData, startingIndex, min(mbSize, source->classData->number - startingIndex) );
    labels = source->labels + startingIndex;
}

//assuming dataList is initiated
void Data::SubDivide(Data** dataList, int numThreads){
    int totalSize = this->totalSize();
    int minimalSize = totalSize / numThreads;
    int addOne = totalSize - minimalSize * numThreads;
    for(int j=0; j<addOne; ++j)
        dataList[j]->SelectMiniBatch(this, j * (minimalSize+1), minimalSize + 1);
    for(int j=addOne; j<numThreads; ++j)
        dataList[j]->SelectMiniBatch(this, addOne * (minimalSize + 1) + (j - addOne) * minimalSize, minimalSize);
}

void Data::SubDivide(Data** dataList, int numThreads, float* velocity){
    float totalVelocity = 0;
    int totalSize = this->totalSize();
    int totalSubSize=0;
    int mbSubSize[numThreads];
    for(int j=0; j<numThreads; ++j)
        totalVelocity+=velocity[j];
    for(int j=0; j<numThreads; ++j){
        mbSubSize[j] = (velocity[j] * totalSize) / totalVelocity;
        totalSubSize += mbSubSize[j];
    }

    if (totalSubSize>totalSize)
        cout<<"ERROR IN SUBDIVISION!!!"<<endl;
//        for(int j=0; j<totalSubSize - totalSize; ++j)
//            if (mbSubSize[j] > 0) --mbSubSize[j];
    //if (totalSubSize<totalSize)
        for(int j=0; j<totalSize - totalSubSize; ++j)
            ++mbSubSize[j];

    int startIndex = 0;
    for(int j=0; j<numThreads; ++j){
        dataList[j]->SelectMiniBatch(this, startIndex, mbSubSize[j]);
        startIndex += mbSubSize[j];
    }
}


int Data::totalSize(){
    return classData->number;
}
//
//void Data::SubDivide(Data** & dataList, int numThreads){
//    int minInd, maxInd;
//    dataList = new Data*[numThreads];
//    for(int j=0; j<numThreads; ++j){
//        dataList[j] = new Data();
//        dataList[j]->classData = new orderedData**[C];
//        dataList[j]->siz = new int[C];
//        dataList[j]->totalSize=0;
//        dataList[j]->C = C;
//
//        for(int c=0; c<C; ++c){
//            minInd = siz[c] / numThreads * j;
//            maxInd = siz[c] / numThreads * (j+1);
//            //cout<<minInd<<" "<<maxInd<<endl;
//            if (j==numThreads-1) maxInd = siz[c];
//            dataList[j]->classData[c] = classData[c] + minInd;
//            dataList[j]->siz[c] = maxInd - minInd;
//            dataList[j]->totalSize += dataList[j]->siz[c];
//        }
//    }
//}


Data::~Data(){
    delete classData;
    delete []labels;
    delete [] siz;
}


//
void DeleteOnlyDataShell(Data* D){
    DeleteOnlyShell(D->classData);
}



//void Data::SelectTranslationInvariantFeatures(Data* source, int maxDist){
//    int vectSize=(1+maxDist)*(1+maxDist)*3+1;
//    computationalNode* MakeInvariant = new TranslationInvariant(maxDist);
//
//    this->AllocateMemory(source->C, source->siz);
//    for(int c=0; c<C; c++)
//        for(int j=0; j<siz[c]; j++){
//            classData[c][j]=new vect(vectSize);
//            MakeInvariant->ForwardPass(source->classData[c][j], classData[c][j]);
//        }
//}
