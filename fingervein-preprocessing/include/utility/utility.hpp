#ifndef OPENFINGER_UTILITY_HPP
#define OPENFINGER_UTILITY_HPP

#include "./dependencies.hpp"

namespace utility {

    class Utility {
        public:
            static inline QImage mat2qImage(const cv::Mat &inputMat, const QImage::Format format) {
                return QImage(inputMat.data, inputMat.cols, inputMat.rows, static_cast<int>(inputMat.step), format);
            }

            static inline cv::Mat qImage2Mat(const QImage &inputQImage, const int format) {
                return cv::Mat(inputQImage.height(), inputQImage.width(), format, const_cast<uchar*>(inputQImage.bits()),
                               static_cast<size_t>(inputQImage.bytesPerLine()));
            }

            static inline cv::Mat array_uchar2mat_uchar(const af::array &inputArray) {
                if (inputArray.type() != u8) {
                    // TODO: replace QDebug with a logging library (e.g. easyLogging++)
                    qDebug() << QString("af::array to cv::Mat: input image is not grayscale.\n");
                    throw std::invalid_argument("Input af::array is of invalid type. Type required: u8");
                }
                return copyAfArrayDataToCvMat<uchar>(inputArray, u8, CV_8UC1);
            }

            static inline cv::Mat array_float2mat_float(const af::array &inputArray) {
                if (inputArray.type() != f32) {
                    // TODO: replace QDebug with a logging library (e.g. easyLogging++)
                    qDebug() << QString("af::array to cv::Mat: input image is not of float type.\n");
                    throw std::invalid_argument("Input af::array is of invalid type. Type required: f32");
                }
                return copyAfArrayDataToCvMat<uchar>(inputArray, f32, CV_32F);
            }

            static inline af::array mat_uchar2array_uchar(const cv::Mat &inputMat) {
                if (!isCvMatType(inputMat, CV_8UC1)) {
                    // TODO: replace QDebug with a logging library (e.g. easyLogging++)
                    qDebug() << QString("OpenCV Mat to AF Array: input image is not grayscale.\n");
                    throw std::invalid_argument("Input cv::Mat is of invalid type. Type required: CV_8UC1");
                }
                cv::Mat transposed;
                cv::transpose(inputMat, transposed);
                return af::array(inputMat.rows, inputMat.cols, transposed.data);
            }

            static inline af::array mat_double2array_double(const cv::Mat &inputMat) {
                if (!isCvMatType(inputMat, CV_64F)) {
                    // TODO: replace QDebug with a logging library (e.g. easyLogging++)
                    qDebug() << QString("OpenCV Mat to AF Array: input image is not of type CV_64F.\n");
                    throw std::invalid_argument("Input cv::Mat is of invalid type. Required type: CV_64F");
                }
                return copyCvMatDataToAfArray<double, double>(inputMat);
            }

            static inline af::array mat_float2array_float(const cv::Mat& inputMat) {
                if (!isCvMatType(inputMat, CV_32F)) {
                    // TODO: replace QDebug with a logging library (e.g. easyLogging++)
                    qDebug() << QString("OpenCV Mat to AF Array: input image is not of type CV_32F.\n");
                    throw std::invalid_argument("Input cv::Mat is of invalid type. Required type: CV_32F");
                }
                return copyCvMatDataToAfArray<float, float>(inputMat);
            }

            static inline af::array mat_uchar2array_float(const cv::Mat& inputMat) {
                if (!isCvMatType(inputMat, CV_8UC1)) {
                    // TODO: replace QDebug with a logging library (e.g. easyLogging++)
                    qDebug() << QString("OpenCV Mat to AF Array: input image is not of type CV_8UC1.\n");
                    throw std::invalid_argument("Input cv::Mat is of invalid type. Required type: CV_8UC1");
                }
                return copyCvMatDataToAfArray<float, uchar>(inputMat);
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

        private:
            static inline bool isCvMatType(const cv::Mat& inputMat, const int type) noexcept {
                return inputMat.type() == type;
            }

            template<typename afType, typename cvType>
            static inline af::array copyCvMatDataToAfArray(const cv::Mat& inputMat) {
                std::vector<afType> imgData(static_cast<unsigned long>(inputMat.rows * inputMat.cols));
                for (int col = 0; inputMat.cols; ++col) {
                    for (int row = 0; inputMat.rows; ++row) {
                        imgData.emplace_back(inputMat.at<cvType>(row, col));
                    }
                }
                return af::array(inputMat.rows, inputMat.cols, imgData.data());
            }

            template<typename afType>
            static inline cv::Mat copyAfArrayDataToCvMat(const af::array& inputArray, const af_dtype inputArrayType, const int cvMatType) {
                auto *afData = inputArray.as(inputArrayType).T().host<afType>();
                cv::Mat outputMat = cv::Mat(static_cast<int>(inputArray.dims(0)),
                                            static_cast<int>(inputArray.dims(1)),
                                            cvMatType, afData);
                af::freeHost(afData);
                return outputMat;
            }
    };
}

#endif //OPENFINGER_UTILITY_HPP
