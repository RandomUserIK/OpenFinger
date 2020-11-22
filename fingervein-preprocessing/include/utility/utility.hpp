#ifndef OPENFINGER_UTILITY_HPP
#define OPENFINGER_UTILITY_HPP

#include "fingervein_preprocessing.hpp"

namespace utility {

    static inline QImage mat2qImage(const cv::Mat &inputMat, const QImage::Format format) {
        return QImage(inputMat.data, inputMat.cols, inputMat.rows, static_cast<int>(inputMat.step), format);
    }

    static inline cv::Mat qImage2Mat(const QImage &inputQImage, const int format) {
        return cv::Mat(inputQImage.height(), inputQImage.width(), format, const_cast<uchar*>(inputQImage.bits()),
                       static_cast<size_t>(inputQImage.bytesPerLine()));
    }

    static inline af::array mat_uchar2array_uchar(const cv::Mat &inputMat) {
        if (inputMat.type() != CV_8UC1) {
            // TODO: replace QDebug with a logging library (e.g. easyLogging++)
            qDebug() << QString("OpenCV Mat to AF Array: input image is not grayscale.\n");
            throw std::invalid_argument("Input cv::Mat is of invalid type. Type required: CV_8UC1");
        }
        cv::Mat transposed;
        cv::transpose(inputMat, transposed);
        return af::array(inputMat.rows, inputMat.cols, transposed.data);
    }

    static inline cv::Mat array_uchar2mat_uchar(const af::array &inputArray) {
        if (inputArray.type() != u8) {
            // TODO: replace QDebug with a logging library (e.g. easyLogging++)
            qDebug() << QString("af::array to cv::Mat: input image is not grayscale.\n");
            throw std::invalid_argument("Input af::array is of invalid type. Type required: u8");
        }
        auto *afData = inputArray.as(u8).T().host<uchar>();
        cv::Mat outputMat = cv::Mat(static_cast<int>(inputArray.dims(0)),
                                    static_cast<int>(inputArray.dims(1)),
                                    CV_8UC1, afData);
        af::freeHost(afData);
        return outputMat;
    }

    static inline af::array mat_double2array_double(const cv::Mat &inputMat) {
        if (inputMat.type() != CV_64F) {
            // TODO: replace QDebug with a logging library (e.g. easyLogging++)
            qDebug() << QString("OpenCV Mat to AF Array: input image is not of type CV_64F.\n");
            throw std::invalid_argument("Input cv::Mat is of invalid type. Required type: CV_64F");
        }

        std::vector<double> imgData(static_cast<unsigned long>(inputMat.rows * inputMat.cols));
        for (int col = 0; inputMat.cols; ++col) {
            for (int row = 0; inputMat.rows; ++row) {
                imgData.emplace_back(inputMat.at<double>(row, col));
            }
        }
        return af::array(inputMat.rows, inputMat.cols, imgData.data());
    }

    static inline af::array mat_float2array_float(const cv::Mat& inputMat) {
        if (inputMat.type() != CV_32F) {
            // TODO: replace QDebug with a logging library (e.g. easyLogging++)
            qDebug() << QString("OpenCV Mat to AF Array: input image is not of type CV_32F.\n");
            throw std::invalid_argument("Input cv::Mat is of invalid type. Required type: CV_32F");
        }

        std::vector<float> imgData(static_cast<unsigned long>(inputMat.rows * inputMat.cols));
        for (int col = 0; inputMat.cols; ++col) {
            for (int row = 0; inputMat.rows; ++row) {
                imgData.emplace_back(inputMat.at<float>(row, col));
            }
        }
        return af::array(inputMat.rows, inputMat.cols, imgData.data());
    }

    static inline af::array mat_uchar2array_float(const cv::Mat& inputMat) {
        if (inputMat.type() != CV_8UC1) {
            // TODO: replace QDebug with a logging library (e.g. easyLogging++)
            qDebug() << QString("OpenCV Mat to AF Array: input image is not of type CV_8UC1.\n");
            throw std::invalid_argument("Input cv::Mat is of invalid type. Required type: CV_8UC1");
        }

        std::vector<float> imgData(static_cast<unsigned long>(inputMat.rows * inputMat.cols));
        for (int col = 0; inputMat.cols; ++col) {
            for (int row = 0; inputMat.rows; ++row) {
                imgData.emplace_back(inputMat.at<uchar>(row, col));
            }
        }
        return af::array(inputMat.rows, inputMat.cols, imgData.data());
    }

    static inline void af_normalizeImage(af::array &inputArray) noexcept {
        af::array min = af::tile(af::min(inputArray),
                                 static_cast<unsigned int>(inputArray.dims(0)),
                                 static_cast<unsigned int>(inputArray.dims(1)));
        af::array max = af::tile(af::max(inputArray),
                                 static_cast<unsigned int>(inputArray.dims(0)),
                                 static_cast<unsigned int>(inputArray.dims(1)));
        inputArray = 255 * ((inputArray.as(f32) - min) / (max - min));
    }
}

#endif //OPENFINGER_UTILITY_HPP
