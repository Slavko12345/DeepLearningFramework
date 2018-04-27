#include "realNumber.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include "mathFunc.h"
#include "vect.h"
using namespace std;

realNumber::realNumber(){
}

realNumber::realNumber(double val): orderedData(1){
    elem[0]=val;
}

int realNumber::Dimensionality(){
    return 0;
}

double& realNumber::At(){
    return elem[0];
}

void realNumber::SetToVectElement(vect* V, int j){
    elem = V->elemLink(j);
    len = 1;
}
