#pragma once

#include <stdlib.h>
#include <time.h>

class RandGenerator {

    const int min, max;

public:
    RandGenerator(const int max = 256, const int min = 0)
        : min(min), max(max)
    {
        //srand(time(NULL));
    }

    int randValue() {
        return rand() % max + min;
    }

};