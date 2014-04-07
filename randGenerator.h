#pragma once

//#include <random>

class RandGenerator {
    //std::mt19937 gen;
    //std::uniform_int_distribution<> dis;
    const int min, max;

public:
    RandGenerator(const int max = 256, const int min = 0)
        : min(min), max(max)
    {
        //std::random_device rd;
        //gen = std::mt19937(rd());
        //dis = std::uniform_int_distribution<>(min, max);
    }

    int rand() {
        return 1; //dis(gen);
    }

};