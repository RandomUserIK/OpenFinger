#ifndef OPENFINGER_PREPROCESSING_HPP
#define OPENFINGER_PREPROCESSING_HPP

#include "opencv2/core.hpp"
#include "QVector"

namespace fingervein {

class Preprocessing {
    public:
        Preprocessing() = default;
        ~Preprocessing() = default;

    private:
        cv::Mat mat;
};

}

#endif //OPENFINGER_PREPROCESSING_HPP
