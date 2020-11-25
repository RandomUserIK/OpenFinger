#ifndef OPENFINGER_STRUCTURES_HPP
#define OPENFINGER_STRUCTURES_HPP

#include "dependencies.hpp"

namespace fingervein {

    // Contrast enhancement
    typedef struct {
        int distance;
        int gaussianBlock;
        int gaussianBorderType;
        double gaussianSigmaXSuace;
        double gaussianSigmaX;
        double gaussianSigmaY;
        cv::Size gaussianKernelSize;
    } ContrastEnhancementParams;

}

#endif //OPENFINGER_STRUCTURES_HPP
