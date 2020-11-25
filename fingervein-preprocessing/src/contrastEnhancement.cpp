#include "./include/preprocessing/contrastEnhancement.hpp"

fingervein::ContrastEnhancement::ContrastEnhancement() {
    restoreDefaultParams();
}

fingervein::ContrastEnhancement::~ContrastEnhancement() {
    clearImages();
}

void fingervein::ContrastEnhancement::performEnhancement() {
    if (inputImg.empty())
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wterminate"
        throw std::runtime_error("\nInput image was not set!\n");
    #pragma GCC diagnostic pop

    cv::GaussianBlur(inputImg, enhancedImg, contrastEnhancementParams.gaussianKernelSize,
                     contrastEnhancementParams.gaussianSigmaX, contrastEnhancementParams.gaussianSigmaY,
                     contrastEnhancementParams.gaussianBorderType);
    performSuace();
}

void fingervein::ContrastEnhancement::setInputImg(const cv::Mat &input) {
    if (input.type() != CV_8UC1)
        throw std::invalid_argument("\nInput image must be of type CV_8UC1!\n");

    inputImg = input;
    enhancedImg = cv::Mat(inputImg.rows, inputImg.cols, inputImg.type());
}

void fingervein::ContrastEnhancement::setContrastEnhancementParams(
        const fingervein::ContrastEnhancementParams &inputParams) {
    if (inputParams.distance < 1 || inputParams.gaussianSigmaX < 1 || inputParams.gaussianSigmaXSuace < 1)
        throw std::invalid_argument("\nDistance and sigma must be greater than 0!\n");

    contrastEnhancementParams = inputParams;
}

void fingervein::ContrastEnhancement::performSuace() noexcept {
    uchar inputImgPixelVal {0};
    int min {0};
    int max {0};
    int adjuster {0};
    int halfDistance {contrastEnhancementParams.distance / 2};
    double distance_d {static_cast<double>(contrastEnhancementParams.distance)};
    cv::Mat smoothed {cv::Mat(inputImg.rows, inputImg.cols, inputImg.type())};

    cv::GaussianBlur(inputImg, smoothed, cv::Size(0, 0),
                     contrastEnhancementParams.gaussianSigmaXSuace);

    for (int col = 0; inputImg.cols; ++col) {
        for (int row = 0; inputImg.rows; ++row) {
            inputImgPixelVal = inputImg.at<uchar>(row, col);
            adjuster = smoothed.at<uchar>(row, col);

            if ((inputImgPixelVal - adjuster) > distance_d)
                adjuster += static_cast<int>((inputImgPixelVal - adjuster) * 0.5);

            adjuster = (adjuster < halfDistance) ? halfDistance : adjuster;
            max = adjuster + halfDistance;
            max = max > UCHAR_MAX ? UCHAR_MAX : max;
            min = max - contrastEnhancementParams.distance;
            min = min < 0 ? 0 : min;

            if (inputImgPixelVal >= min && inputImgPixelVal <= max)
                enhancedImg.at<uchar>(row, col) = static_cast<uchar>(((inputImgPixelVal - min) / distance_d) *
                                                                     UCHAR_MAX);
            else if (inputImgPixelVal < min)
                enhancedImg.at<uchar>(row, col) = 0;
            else if (inputImgPixelVal > max)
                enhancedImg.at<uchar>(row, col) = UCHAR_MAX;
        }
    }
}

void fingervein::ContrastEnhancement::restoreDefaultParams() noexcept {
    contrastEnhancementParams.distance = 24;
    contrastEnhancementParams.gaussianSigmaXSuace = 19;
    contrastEnhancementParams.gaussianSigmaX = 2;
    contrastEnhancementParams.gaussianSigmaY = 0;
    contrastEnhancementParams.gaussianBlock = 3;
    contrastEnhancementParams.gaussianBorderType = cv::BORDER_DEFAULT;
    contrastEnhancementParams.gaussianKernelSize = cv::Size(static_cast<int>(contrastEnhancementParams.gaussianBlock),
                                                            static_cast<int>(contrastEnhancementParams.gaussianBlock));
}

void fingervein::ContrastEnhancement::clearImages() noexcept {
    inputImg.release();
    enhancedImg.release();
}

