#ifndef OPENFINGER_BILATERALFILTERING_HPP
#define OPENFINGER_BILATERALFILTERING_HPP

#include "./dependencies.hpp"
#include "./structures.hpp"

namespace fingervein {

    class BilateralFiltering {
      public:
        BilateralFiltering();
        ~BilateralFiltering();

        cv::Mat applyBilateralFilter();

        void setOriginalImg(const cv::Mat &originalImg) noexcept;
        void setBilateralFilterParams(const BilateralFilterParams &bilateralFilterParams) noexcept;

      private:
        cv::Mat m_originalImg;
        cv::Mat m_blurredImg;

        fingervein::BilateralFilterParams m_bilateralFilterParams {};

        void applyBilateralMultipleTimes() noexcept;
        void clearParams() noexcept;
        void clearResults() noexcept;
    };

} // namespace fingervein

#endif //OPENFINGER_BILATERALFILTERING_HPP
