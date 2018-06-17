#include "randomGenerator.h"
#include "stdlib.h"
#include <iostream>
using namespace std;
#define U_MAX  (0xffffffff)

uint32_t randomGenerator::state = 2;

uint32_t randomGenerator::MaxValue = U_MAX;

void randomGenerator::SetRandomSeed(){
    state = time(NULL) % U_MAX;
}

uint32_t randomGenerator::rand(){
    uint32_t x = state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	state = x;
	return x;
}

float randomGenerator::generateFloat(float maxAbs){
    return (float(randomGenerator::rand()) / float(randomGenerator::MaxValue) - 0.5) * 2.0 * maxAbs;
}

float randomGenerator::generatePositiveFloat(float maxVal){
    return float(randomGenerator::rand()) / float(randomGenerator::MaxValue) * maxVal;
}

bool randomGenerator::generateBool(float p){
    return (randomGenerator::rand() < (p * randomGenerator::MaxValue) );
}
