#include "./include/preprocessing/bilateralFiltering.hpp"

fingervein::BilateralFiltering::BilateralFiltering() {
    m_bilateralFilterParams.diameter = 10;
    m_bilateralFilterParams.borderType = cv::BORDER_DEFAULT;
    m_bilateralFilterParams.timesApplied = 3;
    m_bilateralFilterParams.sigmaColor = 10;
    m_bilateralFilterParams.sigmaSpace = 3;

    clearResults();
}

fingervein::BilateralFiltering::~BilateralFiltering() {
    clearResults();
    clearResults();
}

cv::Mat fingervein::BilateralFiltering::applyBilateralFilter() {
    if (m_bilateralFilterParams.timesApplied < 1)
        throw std::invalid_argument("\nBilateral filter must be applied at least once!\n");

    if (m_bilateralFilterParams.timesApplied == 1) {
        cv::bilateralFilter(m_originalImg, m_blurredImg,
                            m_bilateralFilterParams.diameter,
                            m_bilateralFilterParams.sigmaColor,
                            m_bilateralFilterParams.sigmaSpace,
                            m_bilateralFilterParams.borderType);
        return m_blurredImg;
    }

    applyBilateralMultipleTimes();
    return m_blurredImg;
}

void fingervein::BilateralFiltering::applyBilateralMultipleTimes() noexcept {
    cv::Mat intermediateFirst = cv::Mat(m_originalImg.rows, m_originalImg.cols, m_originalImg.type());
    cv::Mat intermediateSecond = cv::Mat(m_originalImg.rows, m_originalImg.cols, m_originalImg.type());

    intermediateFirst.setTo(0);
    intermediateSecond.setTo(0);

    cv::bilateralFilter(m_originalImg, intermediateFirst,
                        m_bilateralFilterParams.diameter,
                        m_bilateralFilterParams.sigmaColor,
                        m_bilateralFilterParams.sigmaSpace,
                        m_bilateralFilterParams.borderType);

    for (int i = 0; i < m_bilateralFilterParams.timesApplied; ++i) {
        cv::bilateralFilter(intermediateFirst, intermediateSecond,
                            m_bilateralFilterParams.diameter,
                            m_bilateralFilterParams.sigmaColor,
                            m_bilateralFilterParams.sigmaSpace,
                            m_bilateralFilterParams.borderType);

        intermediateFirst.setTo(0);
        intermediateFirst(intermediateSecond);
        intermediateSecond.setTo(0);
    }

    m_blurredImg(intermediateFirst);
    intermediateFirst.release();
    intermediateSecond.release();
}

void fingervein::BilateralFiltering::clearParams() noexcept {
    m_bilateralFilterParams.diameter = 0;
    m_bilateralFilterParams.borderType = cv::BORDER_DEFAULT;
    m_bilateralFilterParams.timesApplied = 1;
    m_bilateralFilterParams.sigmaColor = 0;
    m_bilateralFilterParams.sigmaSpace = 0;
}

void fingervein::BilateralFiltering::clearResults() noexcept {
    m_originalImg.release();
    m_blurredImg.release();
}

void fingervein::BilateralFiltering::setOriginalImg(const cv::Mat &inputImg) noexcept {
    m_originalImg = inputImg;
}

void fingervein::BilateralFiltering::setBilateralFilterParams(
        const fingervein::BilateralFilterParams &inputParams) noexcept {
    m_bilateralFilterParams = inputParams;
}


