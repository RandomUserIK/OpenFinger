#ifndef OPENFINGER_STRUCTURES_HPP
#define OPENFINGER_STRUCTURES_HPP

namespace fingervein {

    // Bilateral filtering
    typedef struct {
        int diameter;
        int borderType;
        int timesApplied;
        double sigmaColor;
        double sigmaSpace;
    } BilateralFilterParams;

}

#endif //OPENFINGER_STRUCTURES_HPP
