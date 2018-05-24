#ifndef __data__
#define __data__

class tensor4D;
class Data{
public:
    int C; //number of classes
    int* siz;
    tensor4D* classData;
    int* labels;

public:
    void Initiate();
    void AllocateMemory(int C_, int number_of_images_, int depth_, int rows_, int cols_);
//    void AllocateMemory(int C_, int mbSize_, int siz_);
//    void AllocateMemory(int C_, int * siz_);
//    void PreAllocateMemory(int C_, int totalSize_);

    void SelectRandomMiniBatch(Data* source, int mbSize);
    void SelectMiniBatch(Data* source, int startingIndex, int mbSize);

    void ReadTrainingMnist();
    void ReadTestMnist();

    void AddCifar10Batch(char batchFileName[], int startingIndex);
    void ReadTrainingCifar10();
    void ReadTestCifar10();

    void SubDivide(Data** dataList, int numThreads);
    void SubDivide(Data** dataList, int numThreads, float* velocity);

    int totalSize();

    ~Data();

    //void SelectTranslationInvariantFeatures(Data* source, int maxDist);
    //void SeparateByClasses();
    //matrix* ReturnNextMiniBatch(int mbSize);
};
void DeleteOnlyDataShell(Data* D);

#endif // __data__

