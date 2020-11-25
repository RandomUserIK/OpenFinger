#ifndef OPENFINGER_BINARIZATION_HPP
#define OPENFINGER_BINARIZATION_HPP

#include "./structures.hpp"

namespace fingervein {

    class Binarization {
        public:
            Binarization();
            ~Binarization();

            void performGaussianBlur() noexcept;

            void performAdaptiveBinarization() noexcept;

            void setInputImg(const cv::Mat &input);

            void setBinarizationParams(const BinarizationParams &inputParams);

        private:
            cv::Mat inputImg;
            cv::Mat binarizedImg;
            fingervein::BinarizationParams binarizationParams;

            void removeBackground() noexcept;
            void clearParams() noexcept;
            void clearResults() noexcept;
    };

}

#endif //OPENFINGER_BINARIZATION_HPP
