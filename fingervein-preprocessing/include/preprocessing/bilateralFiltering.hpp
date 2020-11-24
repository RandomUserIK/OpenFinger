#ifndef OPENFINGER_BILATERALFILTERING_HPP
#define OPENFINGER_BILATERALFILTERING_HPP

#include "./dependencies.hpp"
#include "./structures.hpp"

namespace fingervein {

    class BilateralFiltering {
        public:
            explicit BilateralFiltering();
            ~BilateralFiltering();

            cv::Mat applyBilateralFilter();

            void setOriginalImg(const cv::Mat &originalImg) noexcept;
            void setBilateralFilterParams(const BilateralFilterParams &bilateralFilterParams) noexcept;

        private:
            cv::Mat originalImg;
            cv::Mat blurredImg;

            fingervein::BilateralFilterParams bilateralFilterParams;

            void applyBilateralMultipleTimes() noexcept;
            void clearParams() noexcept;
            void clearResults() noexcept;
    };

}

#endif //OPENFINGER_BILATERALFILTERING_HPP
