#ifndef OPENFINGER_IMAGECONTOUR_HPP
#define OPENFINGER_IMAGECONTOUR_HPP

#include "./dependencies.hpp"

namespace fingervein {

    class ImageContour : public cv::Mat1b {
        public:
            ImageContour() : cv::Mat1b(0, 0) {}

            enum {
                EMPTY = 0,
                CONTOUR = 255,
                INNER = 128
            };

            //////////////////////////////////////////////////////////////////////////////

            //! build frtom the non zero pixels of \arg img, using C4 neigbourhood
            inline void from_image_C4(const cv::Mat1b &img) {
                from_image(img, false);
            }

            //////////////////////////////////////////////////////////////////////////////

            //! build frtom the non zero pixels of \arg img, using C8 neigbourhood
            inline void from_image_C8(const cv::Mat1b &img) {
                from_image(img, true);
            }

            //////////////////////////////////////////////////////////////////////////////

            //! compute contour size
            [[nodiscard]] inline unsigned int contour_size() const {
                return static_cast<unsigned int>(cv::countNonZero(contour_image()));
            }

            //////////////////////////////////////////////////////////////////////////////

            //! compute size of the inner non-zero pixels
            [[nodiscard]] inline unsigned int inside_size() const {
                auto this_cp = *this;
                return static_cast<unsigned int>(cv::countNonZero(this_cp == INNER));
            }

            //////////////////////////////////////////////////////////////////////////////

            //! \return an image where pixels = 255 if on the contour, 0 otherwise
            [[nodiscard]] inline cv::Mat1b contour_image() const {
                auto this_cp = *this;
                #pragma GCC diagnostic ignored "-Wconversion"
                return (this_cp == CONTOUR);
            }

            //////////////////////////////////////////////////////////////////////////////

            //! returns reference to the specified element (2D case)
            inline uchar &operator()(int row, int col) const {
                return data[row * cols + col];
            }

            //////////////////////////////////////////////////////////////////////////////

            //! set a given point (row, col) as empty, with a C4 neighbourhood
            inline void set_point_empty_C4(int row, int col) {
                int key = row * cols + col;
                data[key] = EMPTY;
                if (col && data[key - 1] == INNER) // left
                    data[key - 1] = CONTOUR;
                if (col < colsm && data[key + 1] == INNER) // right
                    data[key + 1] = CONTOUR;
                if (row && data[key - cols] == INNER) // up
                    data[key - cols] = CONTOUR;
                if (row < rowsm && data[key + cols] == INNER) // down
                    data[key + cols] = CONTOUR;
            } // end set_point_empty();

            //////////////////////////////////////////////////////////////////////////////

            //! set a given point (row, col) as empty, with a C8 neighbourhood
            inline void set_point_empty_C8(int row, int col) {
                int key = row * cols + col;
                data[key] = EMPTY;
                bool left_ok = col, right_ok = col < colsm,
                        up_ok = row, down_ok = row < rowsm;
                if (left_ok && up_ok && data[key - 1] == INNER) // left
                    data[key - 1] = CONTOUR;
                if (right_ok && data[key + 1] == INNER) // right
                    data[key + 1] = CONTOUR;
                if (up_ok && data[key - cols] == INNER) // up
                    data[key - cols] = CONTOUR;
                if (down_ok && data[key + cols] == INNER) // down
                    data[key + cols] = CONTOUR;

                // C8
                if (left_ok && up_ok && data[key - 1 - cols] == INNER) // left - up
                    data[key - 1 - cols] = CONTOUR;
                if (right_ok && up_ok && data[key + 1 - cols] == INNER) // right - up
                    data[key + 1 - cols] = CONTOUR;
                if (left_ok && down_ok && data[key - 1 + cols] == INNER) // left - down
                    data[key - 1 + cols] = CONTOUR;
                if (right_ok && down_ok && data[key + 1 + cols] == INNER) // right - down
                    data[key + 1 + cols] = CONTOUR;
            } // end set_point_empty();


            //////////////////////////////////////////////////////////////////////////////

            //! \return a compact string representation of a given ImageContour-linke image
            static std::string to_string(const cv::Mat1b &img) {
                std::ostringstream ans;
                ans << "(" << img.cols << "x" << img.rows << ")" << std::endl;
                for (int row = 0; row < img.rows; ++row) {
                    for (int col = 0; col < img.cols; ++col) {
                        switch (img(row, col)) {
                            case EMPTY:
                                ans << '-';
                                break;
                            case CONTOUR:
                                ans << 'X';
                                break;
                            default:
                            case INNER:
                                ans << 'O';
                                break;
                        } // end switch (img(row, col))
                    }
                    ans << std::endl;
                }
                return ans.str();
            }

            //////////////////////////////////////////////////////////////////////////////

            //! \return a compact string representation of the current contour and inner pixels
            [[nodiscard]] std::string to_string() const {
                return to_string(*this);
            }

            //////////////////////////////////////////////////////////////////////////////

            /*! \return a color image where the contour, the inside and the zero pixels
           * are represented by custom colors */
            const cv::Mat3b &illus(const cv::Vec3b &contour_color = cv::Vec3b(0, 0, 255),
                                   const cv::Vec3b &inner_color = cv::Vec3b(128, 128, 128),
                                   const cv::Vec3b &empty_color = cv::Vec3b(0, 0, 0)) {
                _illus.create(rows, cols);
                auto npixels = static_cast<unsigned int>(cols * rows);
                const uchar *in_ptr = ptr<uchar>(0);
                auto *out_ptr = _illus.ptr<cv::Vec3b>(0);
                for (unsigned int pix_idx = 0; pix_idx < npixels; ++pix_idx) {
                    switch (*in_ptr++) {
                        case EMPTY:
                            *out_ptr = empty_color;
                            break;
                        case CONTOUR:
                            *out_ptr = contour_color;
                            break;
                        default:
                        case INNER:
                            *out_ptr = inner_color;
                            break;
                    } // end switch (control)
                    ++out_ptr;
                } // end loop pix_idx
                return _illus;
            } // end illus()

        private:
            ////////////////////////////////////////////////////////////////////////////////

            inline void from_image(const cv::Mat1b &img, bool C8 = false) {
                // printf("from_image(cols:%i, rows:%i)\n", img.cols, img.rows);
                create(img.rows, img.cols);
                if (cols * rows == 0) {
                    printf("Empty image\n");
                    return;
                }
                assert(img.isContinuous());
                assert(isContinuous());
                setTo(EMPTY);
                colsm = cols - 1;
                rowsm = rows - 1;
                const uchar *img_data = img.data, *img_ptr = img.data;
                auto *out_ptr = ptr<uchar>(0);
                if (C8)
                    from_image_loop_C8(img_data, img_ptr, out_ptr);
                else
                    from_image_loop_C4(img_data, img_ptr, out_ptr);
            }

            ////////////////////////////////////////////////////////////////////////////////

            inline void from_image_loop_C4(const uchar *img_data, const uchar *img_ptr,
                                           uchar *out_ptr) {
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        //printf("row:%i, col:%i\n", row, col);
                        if (*img_ptr) {
                            int key = row * cols + col;
                            if ((!col || col == colsm || !row || row == rowsm) // border
                                || *img_ptr != img_data[key - 1] //left
                                || (col < colsm && *img_ptr != img_data[key + 1]) // right
                                || *img_ptr != img_data[key - cols] // up
                                || (row < rowsm && *img_ptr != img_data[key + cols])) // down
                            { // contour
                                // printf("row:%i, col:%i is contour!\n", row, col);
                                *out_ptr = CONTOUR;
                                //out_ptr[key] = CONTOUR;
                            } // end if real contour
                            else // inner point
                                *out_ptr = INNER;
                            //out_ptr[key] = INNER;
                        } // end if (img_ptr)
                        img_ptr++;
                        out_ptr++;
                    } // end loop col
                } // end loop row
            } // end from_image_loop_C4()

            ////////////////////////////////////////////////////////////////////////////////

            inline void from_image_loop_C8(const uchar *img_data, const uchar *img_ptr,
                                           uchar *out_ptr) {
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        //printf("row:%i, col:%i\n", row, col);
                        if (*img_ptr) {
                            int key = row * cols + col;
                            bool left_ok = col, right_ok = col < colsm,
                                    up_ok = row, down_ok = row < rowsm;
                            if ((!left_ok || !right_ok || !up_ok || !down_ok) // border
                                || *img_ptr != img_data[key - 1] // L
                                || *img_ptr != img_data[key + 1] // R
                                || *img_ptr != img_data[key - cols] // U
                                || *img_ptr != img_data[key + cols] // D
                                || *img_ptr != img_data[key - cols - 1] // LU
                                || *img_ptr != img_data[key + cols - 1] // LD
                                || *img_ptr != img_data[key - cols + 1] // RU
                                || *img_ptr != img_data[key + cols + 1] // RD
                                    ) { // contour
                                // printf("row:%i, col:%i is contour!\n", row, col);
                                *out_ptr = CONTOUR;
                            } // end if real contour
                            else // inner point
                                *out_ptr = INNER;
                        } // end if (img_ptr)
                        img_ptr++;
                        out_ptr++;
                    } // end loop col
                } // end loop row
            } // end from_image_loop_C8()

            //////////////////////////////////////////////////////////////////////////////

            int rowsm {}, colsm {};
            cv::Mat3b _illus;
    };
}

#endif //OPENFINGER_IMAGECONTOUR_HPP
