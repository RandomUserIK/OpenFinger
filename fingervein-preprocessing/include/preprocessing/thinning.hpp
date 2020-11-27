/*!
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/9/11
________________________________________________________________________________
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
________________________________________________________________________________
 */

#ifndef OPENFINGER_THINNING_HPP
#define OPENFINGER_THINNING_HPP

#include "./dependencies.hpp"
#include "./imagecontour.hpp"

namespace fingervein {

    typedef bool (*VoronoiFn)(uchar *skeldata, int iter, int col, int row, int cols);

    class Thinning {
        public:
            bool thinGuoHallFast(const cv::Mat1b &img,
                                 bool inverted,
                                 bool crop_img_before = false,
                                 int max_iters = NOLIMIT);

            cv::Mat getImgSkeleton() const noexcept;

            cv::Mat getImgSkeletonInverted() const noexcept;

        private:
            cv::Mat1b imgSkeleton;
            cv::Mat1b imgSkeletonInverted;

            cv::Rect _bbox;
            fingervein::ImageContour skelcontour;

            std::deque<int> cols_to_set;
            std::deque<int> rows_to_set;
            bool _has_converged;

            bool thin_fast_custom_voronoi_fn(const cv::Mat1b &img, bool inverted, VoronoiFn voronoi_fn,
                                             bool crop_img_before = true, int max_iters = NOLIMIT);

            cv::Rect copy_bounding_box_plusone(const cv::Mat1b &img, cv::Mat1b &out, bool crop_img_before = true);

            template<class _T>
            cv::Rect boundingBox(const cv::Mat_<_T> &img);

            cv::Mat invertColor(const cv::Mat &img);

            static const int NOLIMIT = INT_MAX;
    };
}

#endif //OPENFINGER_THINNING_HPP
