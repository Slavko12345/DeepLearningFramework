#include "activityData.h"
#include <stdlib.h>
#include "randomGenerator.h"
#include <math.h>
#include "mathFunc.h"

activityData::activityData(){
}

activityData::activityData(int len_, float dropRate_): dropRate(dropRate_), len(len_){
    dropping = (dropRate>1E-10);

    activeUnits = new bool[len];
    for(int i=0; i<len; ++i)
        activeUnits[i]=1;
}

void activityData::DropAllExcept(int num){
    this->SetAllNonActive();
    int* remainIndex = new int[num];
    FillRandom(remainIndex, 0, this->len, num);
    for(int j=0; j<num; ++j)
        activeUnits[remainIndex[j] ] = 1;
}

float activityData::dropRateInFact(float dropRate_){
    if (fabs(1.0 - dropRate_)<1E-10)
        return 1.0;

    if (fabs(dropRate_)<1E-10)
        return 0.0;

    if (dropRate_>0.5+1E-8)
        return 1.0 - dropRateInFact(1.0 - dropRate_);

    const int toCompr = round(8 * dropRate_);
    const int toCompr16 = round(16 * dropRate_);

    if (toCompr16 == 1){
        return 0.0625;
    }

    return toCompr / 8.0;
}

void activityData::DropUnits(){
    if (!dropping) return;

    if (fabs(1.0 - dropRate)<1E-10){
        this->SetAllNonActive();
        return;
    }

    if (fabs(dropRate<1E-10)){
        this->SetAllActive();
        return;
    }

    if (dropRate>0.5+1E-8){
        dropRate = 1 - dropRate;
        this->DropUnits();
        this->FlipActivities();
        dropRate = 1 - dropRate;
        return;
    }



    uint32_t gen;
    const int toCompr = round(8 * dropRate);

    const int toCompr16 = round(16 * dropRate);

    if (toCompr16 == 1){
        this->DropUnitsStandard_0_0625();
        return;
    }

    if (toCompr == 1){
        this->DropUnitsStandard_0_125();
        return;
    }

    if (toCompr == 2){
        this->DropUnitsStandard_0_25();
        return;
    }

    if (toCompr == 4){
        this->DropUnitsStandard_0_5();
        return;
    }

    bool* active_10j = activeUnits;
    const int len_10 = len / 10;

    for(int j=0; j<len_10; ++j){
        gen = randomGenerator::rand();

        for(int d=0; d<10; ++d){
            active_10j[d] = ( (gen & 7) >= toCompr );
            gen >>= 3;
        }
        active_10j += 10;
    }

    const int remain = len - len_10 * 10;
    if (remain > 0){
        gen = randomGenerator::rand();
        for(int d=0; d<remain; ++d){
            active_10j[d] = ( (gen & 7) >= toCompr );
            gen >>= 3;
        }
    }
}


void activityData::DropUnitsStandard_0_0625(){
    uint32_t gen;

    bool* active_8j = activeUnits;
    const int len_8 = len / 8;

    for(int j=0; j<len_8; ++j){
        gen = randomGenerator::rand();

        for(int d=0; d<8; ++d){
            active_8j[d] = (gen & 15);
            gen >>= 4;
        }
        active_8j += 8;
    }

    const int remain = len - len_8 * 8;
    if (remain > 0){
        gen = randomGenerator::rand();
        for(int d=0; d<remain; ++d){
            active_8j[d] = (gen & 15);
            gen >>= 4;
        }
    }

}



void activityData::DropUnitsStandard_0_125(){
    uint32_t gen;

    bool* active_10j = activeUnits;
    const int len_10 = len / 10;

    for(int j=0; j<len_10; ++j){
        gen = randomGenerator::rand();

        for(int d=0; d<10; ++d){
            active_10j[d] = (gen & 7);
            gen >>= 3;
        }
        active_10j += 10;
    }

    const int remain = len - len_10 * 10;
    if (remain > 0){
        gen = randomGenerator::rand();
        for(int d=0; d<remain; ++d){
            active_10j[d] = (gen & 7);
            gen >>= 3;
        }
    }

}



void activityData::DropUnitsStandard_0_25(){
    uint32_t gen;

    bool* active_16j = activeUnits;
    const int len_16 = len / 16;

    for(int j=0; j<len_16; ++j){
        gen = randomGenerator::rand();

        for(int d=0; d<16; ++d){
            active_16j[d] = (gen & 3);
            gen >>= 2;
        }
        active_16j += 16;
    }

    const int remain = len - len_16 * 16;
    if (remain > 0){
        gen = randomGenerator::rand();
        for(int d=0; d<remain; ++d){
            active_16j[d] = (gen & 3);
            gen >>= 2;
        }
    }

}



void activityData::DropUnitsStandard_0_5(){
    uint32_t gen;

    bool* active_32j = activeUnits;
    const int len_32 = len / 32;

    for(int j=0; j<len_32; ++j){
        gen = randomGenerator::rand();

        for(int d=0; d<32; ++d){
            active_32j[d] = (gen & 1);
            gen >>= 1;
        }
        active_32j += 32;
    }

    const int remain = len - len_32 * 32;
    if (remain > 0){
        gen = randomGenerator::rand();
        for(int d=0; d<remain; ++d){
            active_32j[d] = (gen & 1);
            gen >>= 1;
        }
    }
}



void activityData::SetAllActive(){
    if (!dropping) return;
    for(int j=0; j<len; j++)
        activeUnits[j]=1;
}

void activityData::SetAllNonActive(){
    if (!dropping) return;
    for(int j=0; j<len; j++)
        activeUnits[j]=0;
}

void activityData::PrintActivities(){
    for(int j=0; j<len; j++)
        cout<<activeUnits[j];
    cout<<endl;
}

void activityData::FlipActivities(){
    for(int j=0; j<len; ++j)
        activeUnits[j] = !activeUnits[j];
}

float activityData::ActiveProportion(){
    int active = 0;
    for(int j=0; j<len; ++j)
        active += activeUnits[j];
    return float(active) / len;
}

int activityData::ActiveLen(){
    int active = 0;
    for(int j=0; j<len; ++j)
        active += activeUnits[j];
    return active;
}

void activityData::SubActivityData(activityData* act, int startIndex_, int len_){
    dropping = act->dropping;
    dropRate = act->dropRate;
    len = len_;
    activeUnits = act->activeUnits + startIndex_;
}



activityData::~activityData(){
    delete[] activeUnits;
}
