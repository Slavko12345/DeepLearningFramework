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

double randomGenerator::generateDouble(double maxAbs){
    return (double(randomGenerator::rand()) / double(randomGenerator::MaxValue) - 0.5) * 2.0 * maxAbs;
}

double randomGenerator::generatePositiveDouble(double maxVal){
    return double(randomGenerator::rand()) / double(randomGenerator::MaxValue) * maxVal;
}

bool randomGenerator::generateBool(double p){
    return (randomGenerator::rand() < (p * randomGenerator::MaxValue) );
}
