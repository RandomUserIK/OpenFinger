#include "./include/preprocessing/binarization.hpp"

fingervein::Binarization::Binarization() {
    // TODO: add threshold params
    binarizationParams.holeSize = 20;
    binarizationParams.adaptiveBlockSize = 47;
    binarizationParams.adaptiveMaxValue = 255;
    binarizationParams.adaptiveMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    binarizationParams.adaptiveType = 0;
    binarizationParams.adaptiveC = 1;
    binarizationParams.gaussianBorderType = cv::BORDER_DEFAULT;
    binarizationParams.gaussianSigmaX = 1;
    binarizationParams.gaussianSigmaY = 0;
    binarizationParams.gaussianKernelSize = cv::Size(3,3);
}

fingervein::Binarization::~Binarization() {
    clearParams();
    clearResults();
}

void fingervein::Binarization::setInputImg(const cv::Mat &input) {
    inputImg = input;
    binarizedImg = cv::Mat(inputImg.rows, inputImg.cols, inputImg.type());
}

void fingervein::Binarization::setBinarizationParams(const fingervein::BinarizationParams &inputParams) {
    // TODO: input check
    binarizationParams = inputParams;
}

void fingervein::Binarization::performGaussianBlur() noexcept {

}

void fingervein::Binarization::performAdaptiveBinarization() noexcept {

}

void fingervein::Binarization::removeBackground() noexcept {

}

void fingervein::Binarization::clearParams() noexcept {
    // TODO: clear or restore defaults?
    binarizationParams.holeSize = 0;
    binarizationParams.adaptiveBlockSize = 0;
    binarizationParams.adaptiveMaxValue = 0;
    binarizationParams.adaptiveMethod = 0;
    binarizationParams.adaptiveType = 0;
    binarizationParams.adaptiveC = 0;
    binarizationParams.gaussianBorderType = 0;
    binarizationParams.gaussianSigmaX = 0;
    binarizationParams.gaussianSigmaY = 0;
    binarizationParams.gaussianKernelSize = cv::Size(0,0);
}

void fingervein::Binarization::clearResults() noexcept {
    // TODO: find a more appropriate name for this method
    inputImg.release();
    binarizedImg.release();
}

