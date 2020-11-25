#ifndef OPENFINGER_STRUCTURES_HPP
#define OPENFINGER_STRUCTURES_HPP

#include "dependencies.hpp"

namespace fingervein {


    // Binarization
    typedef struct {
        int holeSize;
        int adaptiveBlockSize;
        int adaptiveType;
        int adaptiveMethod;
        int gaussianBorderType;
        int thresholdType;
        cv::Size gaussianKernelSize;
        double gaussianSigmaX;
        double gaussianSigmaY;
        double adaptiveC;
        double adaptiveMaxValue;
        double thresholdValue;
        double thresholdMaxValue;
    } BinarizationParams;
}

#endif //OPENFINGER_STRUCTURES_HPP
