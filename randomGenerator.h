#ifndef __randomGenerator__
#define __randomGenerator__
#include <stdint.h>

struct randomGenerator{
    static uint32_t state;
    static void SetRandomSeed();
    static uint32_t rand();
    static uint32_t MaxValue;
    static double generateDouble(double maxAbs);
    static double generatePositiveDouble(double maxVal);
    static bool generateBool(double p); //p - probability of 1
};

#endif // __randomGenerator__
