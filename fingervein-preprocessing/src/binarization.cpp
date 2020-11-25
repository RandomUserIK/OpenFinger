#include "./include/preprocessing/binarization.hpp"

fingervein::Binarization::Binarization() {
    restoreDefaultParams();
}

fingervein::Binarization::~Binarization() {
    clearImages();
}

void fingervein::Binarization::setInputImg(const cv::Mat &input) {
    if (!inputImg.empty() || !binarizedImg.empty())
        clearImages();

    inputImg = input;
    binarizedImg = cv::Mat(inputImg.rows, inputImg.cols, inputImg.type());
}

void fingervein::Binarization::setBinarizationParams(const fingervein::BinarizationParams &inputParams) {
    binarizationParams = inputParams;
}

void fingervein::Binarization::performGaussianBlur() {
    if (inputImg.empty())
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wterminate"
        throw std::runtime_error("Input image was not set!\n");
        #pragma GCC diagnostic pop

    cv::GaussianBlur(inputImg, binarizedImg, binarizationParams.gaussianKernelSize,
                     binarizationParams.gaussianSigmaX, binarizationParams.gaussianSigmaY,
                     binarizationParams.gaussianBorderType);
    cv::threshold(binarizedImg, binarizedImg, binarizationParams.thresholdValue,
                  binarizationParams.thresholdMaxValue, binarizationParams.thresholdType);
}

void fingervein::Binarization::performAdaptiveBinarization() {
    if (inputImg.empty())
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wterminate"
        throw std::runtime_error("Input image was not set!\n");
        #pragma GCC diagnostic pop

    cv::adaptiveThreshold(inputImg, binarizedImg, binarizationParams.adaptiveMaxValue,
                          binarizationParams.adaptiveMethod,binarizationParams.adaptiveType,
                          binarizationParams.adaptiveBlockSize, binarizationParams.adaptiveC);
}

void fingervein::Binarization::restoreDefaultParams() noexcept {
    binarizationParams.holeSize = 20;
    binarizationParams.adaptiveBlockSize = 47;
    binarizationParams.adaptiveMaxValue = 255;
    binarizationParams.adaptiveMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    binarizationParams.adaptiveType = 0;
    binarizationParams.adaptiveC = 1;
    binarizationParams.gaussianBorderType = cv::BORDER_DEFAULT;
    binarizationParams.gaussianSigmaX = 1;
    binarizationParams.gaussianSigmaY = 0;
    binarizationParams.gaussianKernelSize = cv::Size(3, 3);
    binarizationParams.thresholdMaxValue = 255;
    binarizationParams.thresholdValue = 0;
    binarizationParams.thresholdType = cv::THRESH_BINARY + cv::THRESH_OTSU;
}

void fingervein::Binarization::clearImages() noexcept {
    inputImg.release();
    binarizedImg.release();
}

