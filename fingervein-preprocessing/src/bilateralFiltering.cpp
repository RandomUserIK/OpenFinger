#include "./include/preprocessing/bilateralFiltering.hpp"

fingervein::BilateralFiltering::BilateralFiltering() {
    bilateralFilterParams.diameter = 10;
    bilateralFilterParams.borderType = cv::BORDER_DEFAULT;
    bilateralFilterParams.timesApplied = 3;
    bilateralFilterParams.sigmaColor = 10;
    bilateralFilterParams.sigmaSpace = 3;

    clearResults();
}

fingervein::BilateralFiltering::~BilateralFiltering() {
    clearResults();
    clearResults();
}

cv::Mat fingervein::BilateralFiltering::applyBilateralFilter() {
    if (bilateralFilterParams.timesApplied < 1)
        throw std::invalid_argument("\nBilateral filter must be applied at least once!\n");

    if (bilateralFilterParams.timesApplied == 1) {
        cv::bilateralFilter(originalImg, blurredImg,
                            bilateralFilterParams.diameter,
                            bilateralFilterParams.sigmaColor,
                            bilateralFilterParams.sigmaSpace,
                            bilateralFilterParams.borderType);
        return blurredImg;
    }

    applyBilateralMultipleTimes();
    return blurredImg;
}

void fingervein::BilateralFiltering::applyBilateralMultipleTimes() noexcept {
    cv::Mat intermediateFirst = cv::Mat(originalImg.rows, originalImg.cols, originalImg.type());
    cv::Mat intermediateSecond = cv::Mat(originalImg.rows, originalImg.cols, originalImg.type());

    intermediateFirst.setTo(0);
    intermediateSecond.setTo(0);

    cv::bilateralFilter(originalImg, intermediateFirst,
                        bilateralFilterParams.diameter,
                        bilateralFilterParams.sigmaColor,
                        bilateralFilterParams.sigmaSpace,
                        bilateralFilterParams.borderType);

    for (int i = 0; i < bilateralFilterParams.timesApplied; ++i) {
        cv::bilateralFilter(intermediateFirst, intermediateSecond,
                            bilateralFilterParams.diameter,
                            bilateralFilterParams.sigmaColor,
                            bilateralFilterParams.sigmaSpace,
                            bilateralFilterParams.borderType);

        intermediateFirst.setTo(0);
        intermediateFirst(intermediateSecond);
        intermediateSecond.setTo(0);
    }

    blurredImg(intermediateFirst);
    intermediateFirst.release();
    intermediateSecond.release();
}

void fingervein::BilateralFiltering::clearParams() noexcept {
    bilateralFilterParams.diameter = 0;
    bilateralFilterParams.borderType = cv::BORDER_DEFAULT;
    bilateralFilterParams.timesApplied = 1;
    bilateralFilterParams.sigmaColor = 0;
    bilateralFilterParams.sigmaSpace = 0;
}

void fingervein::BilateralFiltering::clearResults() noexcept {
    originalImg.release();
    blurredImg.release();
}

void fingervein::BilateralFiltering::setOriginalImg(const cv::Mat &inputImg) noexcept {
    originalImg = inputImg;
}

void fingervein::BilateralFiltering::setBilateralFilterParams(
        const fingervein::BilateralFilterParams &inputParams) noexcept {
    bilateralFilterParams = inputParams;
}


