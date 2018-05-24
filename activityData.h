#ifndef __activityData__
#define __activityData__
#include <vector>
#include <iostream>
using namespace std;


struct activityData{
    bool dropping;
    double dropRate;
    int len;
    bool* activeUnits;

    activityData();
    activityData(int len_, double dropRate_);
    void DropUnits();
    void DropUnitsStandard_0_0625();
    void DropUnitsStandard_0_125();
    void DropUnitsStandard_0_25();
    void DropUnitsStandard_0_5();
    void DropAllExcept(int num);

    void SetAllActive();
    void SetAllNonActive();
    void PrintActivities();
    void FlipActivities();
    double ActiveProportion();
    int ActiveLen();
    void SubActivityData(activityData* act, int startIndex_, int len_);
    static double dropRateInFact(double dropRate_);
    ~activityData();
};

#endif // __activityData__
