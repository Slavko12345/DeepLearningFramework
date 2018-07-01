#ifndef __activityData__
#define __activityData__
#include <vector>
#include <iostream>
using namespace std;


struct activityData{
    bool dropping;
    float dropRate;
    int len;
    bool* activeUnits;

    activityData();
    activityData(int len_, float dropRate_);
    void DropUnits();
    void DropUnitsStandard_0_0625();
    void DropUnitsStandard_0_125();
    void DropUnitsStandard_0_25();
    void DropUnitsStandard_0_5();
    void DropAllExcept(int num);
    void Drop_2_2(int remainNum = 1);


    void SetAllActive();
    void SetAllNonActive();
    void PrintActivities();
    void PrintActivitiesAsMatrix();
    void FlipActivities();
    float ActiveProportion();
    int ActiveLen();
    void SubActivityData(activityData* act, int startIndex_, int len_);
    static float dropRateInFact(float dropRate_);
    ~activityData();
};

#endif // __activityData__
