/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

/****************************************************************************************\
*                                       Image Alignment (ECC algorithm)                  *
\****************************************************************************************/

using namespace cv;

static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, const Mat& src5,
                                    Mat& dst) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert(src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols * 8));
    CV_Assert(dst.type() == CV_MAKETYPE(CV_32F, src1.channels()));

    CV_Assert(src5.isContinuous());

    const float* hptr = src5.ptr<float>(0);

    const float h0_ = hptr[0];
    const float h1_ = hptr[3];
    const float h2_ = hptr[6];
    const float h3_ = hptr[1];
    const float h4_ = hptr[4];
    const float h5_ = hptr[7];
    const float h6_ = hptr[2];
    const float h7_ = hptr[5];

    const int w = src1.cols;

    // create denominator for all points as a block
    Mat den_;
    addWeighted(src3, h2_, src4, h5_, 1.0, den_);

    // create projected points
    Mat hatX_, hatY_;
    addWeighted(src3, h0_, src4, h3_, 0.0, hatX_);
    hatX_ += h6_;

    addWeighted(src3, h1_, src4, h4_, 0.0, hatY_);
    hatY_ += h7_;

    divide(-hatY_, den_, hatY_);
    divide(-hatX_, den_, hatX_);

    // instead of dividing each block with den,
    // just pre-divide the block of gradients (it's more efficient)

    Mat src1Divided_;
    Mat src2Divided_;

    divide(src1, den_, src1Divided_);
    divide(src2, den_, src2Divided_);

    // compute Jacobian blocks (8 blocks)

    dst.colRange(0, w) = src1Divided_.mul(src3);  // 1

    dst.colRange(w, 2 * w) = src2Divided_.mul(src3);  // 2

    Mat temp_ = (hatX_.mul(src1Divided_) + hatY_.mul(src2Divided_));
    dst.colRange(2 * w, 3 * w) = temp_.mul(src3);  // 3

    hatX_.release();
    hatY_.release();

    dst.colRange(3 * w, 4 * w) = src1Divided_.mul(src4);  // 4

    dst.colRange(4 * w, 5 * w) = src2Divided_.mul(src4);  // 5

    dst.colRange(5 * w, 6 * w) = temp_.mul(src4);  // 6

    src1Divided_.copyTo(dst.colRange(6 * w, 7 * w));  // 7

    src2Divided_.copyTo(dst.colRange(7 * w, 8 * w));  // 8
}

static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4,
                                         const Mat& src5, Mat& dst) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert(src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols * 3));
    CV_Assert(dst.type() == CV_MAKETYPE(CV_32F, src1.channels()));

    CV_Assert(src5.isContinuous());

    const float* hptr = src5.ptr<float>(0);

    const float h0 = hptr[0];  // cos(theta)
    const float h1 = hptr[3];  // sin(theta)

    const int w = src1.cols;

    // create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    Mat hatX = -(src3 * h1) - (src4 * h0);

    // create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    Mat hatY = (src3 * h0) - (src4 * h1);

    // compute Jacobian blocks (3 blocks)
    dst.colRange(0, w) = (src1.mul(hatX)) + (src2.mul(hatY));  // 1

    src1.copyTo(dst.colRange(w, 2 * w));      // 2
    src2.copyTo(dst.colRange(2 * w, 3 * w));  // 3
}

static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2, const Mat& src3, const Mat& src4, Mat& dst) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert(src1.rows == dst.rows);
    CV_Assert(dst.cols == (6 * src1.cols));

    CV_Assert(dst.type() == CV_MAKETYPE(CV_32F, src1.channels()));

    const int w = src1.cols;

    // compute Jacobian blocks (6 blocks)

    dst.colRange(0, w) = src1.mul(src3);          // 1
    dst.colRange(w, 2 * w) = src2.mul(src3);      // 2
    dst.colRange(2 * w, 3 * w) = src1.mul(src4);  // 3
    dst.colRange(3 * w, 4 * w) = src2.mul(src4);  // 4
    src1.copyTo(dst.colRange(4 * w, 5 * w));      // 5
    src2.copyTo(dst.colRange(5 * w, 6 * w));      // 6
}

static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst) {
    CV_Assert(src1.size() == src2.size());

    CV_Assert(src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols * 2));
    CV_Assert(dst.type() == CV_MAKETYPE(CV_32F, src1.channels()));

    const int w = src1.cols;

    // compute Jacobian blocks (2 blocks)
    src1.copyTo(dst.colRange(0, w));
    src2.copyTo(dst.colRange(w, 2 * w));
}

static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst) {
    /* this functions is used for two types of projections. If src1.cols ==src.cols
    it does a blockwise multiplication (like in the outer product of vectors)
    of the blocks in matrices src1 and src2 and dst
    has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    The number_of_blocks is equal to the number of parameters we are lloking for
    (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

    */
    CV_Assert(src1.rows == src2.rows);
    CV_Assert((src1.cols % src2.cols) == 0);
    int w;

    float* dstPtr = dst.ptr<float>(0);

    if (src1.cols != src2.cols) {  // dst.cols==1
        w = src2.cols;
        for (int i = 0; i < dst.rows; i++) {
            dstPtr[i] = (float)src2.dot(src1.colRange(i * w, (i + 1) * w));
        }
    }

    else {
        CV_Assert(dst.cols == dst.rows);  // dst is square (and symmetric)
        w = src2.cols / dst.cols;
        Mat mat;
        for (int i = 0; i < dst.rows; i++) {
            mat = Mat(src1.colRange(i * w, (i + 1) * w));
            dstPtr[i * (dst.rows + 1)] = (float)std::pow(norm(mat), 2);  // diagonal elements

            for (int j = i + 1; j < dst.cols; j++) {  // j starts from i+1
                dstPtr[i * dst.cols + j] = (float)mat.dot(src2.colRange(j * w, (j + 1) * w));
                dstPtr[j * dst.cols + i] = dstPtr[i * dst.cols + j];  // due to symmetry
            }
        }
    }
}

static void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType) {
    CV_Assert(map_matrix.type() == CV_32FC1);
    CV_Assert(update.type() == CV_32FC1);

    CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN || motionType == MOTION_AFFINE ||
              motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert(map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert(map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert(map_matrix.rows == 2 && update.rows == 2);

    CV_Assert(update.cols == 1);

    CV_Assert(map_matrix.isContinuous());
    CV_Assert(update.isContinuous());

    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);

    if (motionType == MOTION_TRANSLATION) {
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float)cos(new_theta);
        mapPtr[3] = (float)sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}

/** Function that computes enhanced corelation coefficient from Georgios et.al. 2008
 *   See https://github.com/opencv/opencv/issues/12432
 */
double cv::computeECC(InputArray templateImage, InputArray inputImage, InputArray inputMask) {
    CV_Assert(!templateImage.empty());
    CV_Assert(!inputImage.empty());

    CV_Assert(templateImage.channels() == 1 || templateImage.channels() == 3);

    if (!(templateImage.type() == inputImage.type()))
        CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

    Scalar meanTemplate, sdTemplate;

    int active_pixels = inputMask.empty() ? templateImage.size().area() : countNonZero(inputMask);
    int type = templateImage.type();
    meanStdDev(templateImage, meanTemplate, sdTemplate, inputMask);
    Mat templateImage_zeromean = Mat::zeros(templateImage.size(), templateImage.type());
    Mat templateMat = templateImage.getMat();
    Mat inputMat = inputImage.getMat();

    /*
     * For unsigned ints, when the mean is computed and subtracted, any values less than the mean
     * will be set to 0 (since there are no negatives values). This impacts the norm and dot product, which
     * ultimately results in an incorrect ECC. To circumvent this problem, if unsigned ints are provided,
     * we convert them to a signed ints with larger resolution for the subtraction step.
     */
    if (type == CV_8U || type == CV_16U) {
        int newType = type == CV_8U ? CV_16S : CV_32S;
        Mat templateMatConverted, inputMatConverted;
        templateMat.convertTo(templateMatConverted, newType);
        cv::swap(templateMat, templateMatConverted);
        inputMat.convertTo(inputMatConverted, newType);
        cv::swap(inputMat, inputMatConverted);
    }
    subtract(templateMat, meanTemplate, templateImage_zeromean, inputMask);
    double templateImagenorm = std::sqrt(active_pixels * cv::norm(sdTemplate, NORM_L2SQR));

    Scalar meanInput, sdInput;

    Mat inputImage_zeromean = Mat::zeros(inputImage.size(), inputImage.type());
    meanStdDev(inputImage, meanInput, sdInput, inputMask);
    subtract(inputMat, meanInput, inputImage_zeromean, inputMask);
    double inputImagenorm = std::sqrt(active_pixels * norm(sdInput, NORM_L2SQR));

    return templateImage_zeromean.dot(inputImage_zeromean) / (templateImagenorm * inputImagenorm);
}


double cv::findTransformECCWithMask( InputArray templateImage,
                                 InputArray inputImage,
                                 InputArray templateMask,
                                 InputArray inputMask,
                                 InputOutputArray warpMatrix,
                                 int motionType,
                                 TermCriteria criteria,
                                 int gaussFiltSize) {
    Mat src = templateImage.getMat();  // template image
    Mat dst = inputImage.getMat();     // input image (to be warped)
    Mat map = warpMatrix.getMat();     // warp (transformation)

    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());

    CV_Assert(src.channels() == 1 || src.channels() == 3);
    CV_Assert(src.channels() == dst.channels());
    CV_Assert(src.depth() == dst.depth());
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F || src.depth() == CV_64F);

    // If the user passed an un-initialized warpMatrix, initialize to identity
    if (map.empty()) {
        int rowCount = 2;
        if (motionType == MOTION_HOMOGRAPHY)
            rowCount = 3;

        warpMatrix.create(rowCount, 3, CV_32FC1);
        map = warpMatrix.getMat();
        map = Mat::eye(rowCount, 3, CV_32F);
    }

    if (!(src.type() == dst.type()))
        CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

    if (map.type() != CV_32FC1)
        CV_Error(Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

    CV_Assert(map.cols == 3);
    CV_Assert(map.rows == 2 || map.rows == 3);

    CV_Assert(motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY || motionType == MOTION_EUCLIDEAN ||
              motionType == MOTION_TRANSLATION);

    if (motionType == MOTION_HOMOGRAPHY) {
        CV_Assert(map.rows == 3);
    }

    CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

    int paramTemp = 6;  // default: affine
    switch (motionType) {
        case MOTION_TRANSLATION:
            paramTemp = 2;
            break;
        case MOTION_EUCLIDEAN:
            paramTemp = 3;
            break;
        case MOTION_HOMOGRAPHY:
            paramTemp = 8;
            break;
    }

    const int numberOfParameters = paramTemp;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);

    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
    for (j = 0; j < ws; j++) XcoPtr[j] = (float)j;
    for (j = 0; j < hs; j++) YcoPtr[j] = (float)j;

    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);

    Xcoord.release();
    Ycoord.release();

    const int channels = src.channels();
    int type = CV_MAKETYPE(CV_32F, channels);

    std::vector<cv::Mat> XgridCh(channels, Xgrid);
    cv::merge(XgridCh, Xgrid);

    std::vector<cv::Mat> YgridCh(channels, Ygrid);
    cv::merge(YgridCh, Ygrid);

    Mat templateZM = Mat(hs, ws, type);     // to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, type);  // to store the (smoothed) template
    Mat imageFloat = Mat(hd, wd, type);     // to store the (smoothed) input image
    Mat imageWarped = Mat(hs, ws, type);    // to store the warped zero-mean input image
    Mat imageMask = Mat(hs, ws, CV_8U);     // to store the final mask

    // Gaussian filtering is optional
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // needed matrices for gradients and warped gradients
    Mat gradientX = Mat::zeros(hd, wd, type);
    Mat gradientY = Mat::zeros(hd, wd, type);
    Mat gradientXWarped = Mat(hs, ws, type);
    Mat gradientYWarped = Mat(hs, ws, type);

    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    filter2D(imageFloat, gradientX, -1, dx);
    filter2D(imageFloat, gradientY, -1, dx.t());

    // To use in mask warping
    Mat templtMask;
    if(templateMask.empty())
    {
        templtMask = Mat::ones(hs, ws, CV_8U);
    }
    else
    {
        threshold(templateMask, templtMask, 0, 1, THRESH_BINARY);
        templtMask.convertTo(templtMask, CV_32F);
        GaussianBlur(templtMask, templtMask, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        templtMask *= (0.5/0.95);
        templtMask.convertTo(templtMask, CV_8U);
    }

    //to use it for mask warping
    Mat preMask;
    if(inputMask.empty())
    {
        preMask = Mat::ones(hd, wd, CV_8U);
    }
    else
    {
        Mat preMaskFloat;
        threshold(inputMask, preMask, 0, 1, THRESH_BINARY);

        preMask.convertTo(preMaskFloat, CV_32F);
        GaussianBlur(preMaskFloat, preMaskFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        // Change threshold.
        preMaskFloat *= (0.5/0.95);
        // Rounding conversion.
        preMaskFloat.convertTo(preMask, CV_8U);

        // If there's no template mask, we can apply image masks to gradients only once.
        // Otherwise, we'll need to combine the template and image masks at each iteration.
        if (templateMask.empty())
        {
            cv::Mat zeroMask = (preMask == 0);
            gradientX.setTo(0, zeroMask);
            gradientY.setTo(0, zeroMask);
        }
    }

    // matrices needed for solving linear equation system for maximizing ECC
    Mat jacobian = Mat(hs, ws * numberOfParameters, type);
    Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);  // transformation parameter correction
    Mat error = Mat(hs, ws, CV_32F);                  // error as 2D matrix

    const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
    const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;

    // iteratively update map_matrix
    double rho = -1;
    double last_rho = -termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++) {
        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        if (motionType != MOTION_HOMOGRAPHY) {
            warpAffine(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            warpAffine(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpAffine(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpAffine(preMask, imageMask, map, imageMask.size(), maskFlags);
        } else {
            warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            warpPerspective(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpPerspective(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpPerspective(preMask, imageMask, map, imageMask.size(), maskFlags);
        }

        if (!templateMask.empty())
        {
            cv::bitwise_and(imageMask, templtMask, imageMask);

            cv::Mat zeroMask = (imageMask == 0);
            gradientXWarped.setTo(0, zeroMask);
            gradientYWarped.setTo(0, zeroMask);
        }

        Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped, imgMean, imgStd, imageMask);
        meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

        subtract(imageWarped, imgMean, imageWarped, imageMask);  // zero-mean input
        templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
        subtract(templateFloat, tmpMean, templateZM, imageMask);  // zero-mean template

        int validPixels = countNonZero(imageMask);
        double tmpNorm = std::sqrt(validPixels * cv::norm(tmpStd, cv::NORM_L2SQR));
        double imgNorm = std::sqrt(validPixels * cv::norm(imgStd, cv::NORM_L2SQR));

        // calculate jacobian of image wrt parameters
        switch (motionType) {
            case MOTION_AFFINE:
                image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
                break;
            case MOTION_HOMOGRAPHY:
                image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
            case MOTION_TRANSLATION:
                image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
                break;
            case MOTION_EUCLIDEAN:
                image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
        }

        // calculate Hessian and its inverse
        project_onto_jacobian_ECC(jacobian, jacobian, hessian);

        hessianInv = hessian.inv();

        const double correlation = templateZM.dot(imageWarped);

        // calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho;
        rho = correlation / (imgNorm * tmpNorm);
        if (cvIsNaN(rho)) {
            CV_Error(Error::StsNoConv, "NaN encountered.");
        }

        // project images into jacobian
        project_onto_jacobian_ECC(jacobian, imageWarped, imageProjection);
        project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);

        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv * imageProjection;
        const double lambda_n = (imgNorm * imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0) {
            rho = -1;
            CV_Error(Error::StsNoConv,
                     "The algorithm stopped before its convergence. The correlation is going to be minimized. Images "
                     "may be uncorrelated or non-overlapped");
        }
        const double lambda = (lambda_n / lambda_d);

        // estimate the update step delta_p
        error = lambda * templateZM - imageWarped;
        project_onto_jacobian_ECC(jacobian, error, errorProjection);
        deltaP = hessianInv * errorProjection;

        // update warping matrix
        update_warping_matrix_ECC(map, deltaP, motionType);
    }

    // return final correlation coefficient
    return rho;
}

double cv::findTransformECC(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria,
                            InputArray inputMask,
                            int gaussFiltSize
                            ) {
    return findTransformECCWithMask(templateImage, inputImage, noArray(), inputMask,
            warpMatrix, motionType, criteria, gaussFiltSize);
}

double cv::findTransformECC(InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix,
                            int motionType, TermCriteria criteria, InputArray inputMask) {
    // Use default value of 5 for gaussFiltSize to maintain backward compatibility.
    return findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, 5);
}
// =============================== PYRAMIDAL VERSION OF GRAYSCALE ECC ================================
template<int motionType> struct MotionTraits {};

template<> struct MotionTraits<MOTION_TRANSLATION> {
    enum { paramAmount = 2 };
    static inline void tail_handler_get_coord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col, 
                                              float numeratorX0,
                                              float numeratorY0,
                                              float /*denominator0*/,
                                              float /*a00*/,
                                              float /*a10*/,
                                              float /*a20*/)
    {
        denominator = 0;
        sx = (numeratorX0 + col);
        sy = numeratorY0;
    }
    template<typename elemtype> 
    static constexpr std::array<float, paramAmount> fillJacobian(int /*col*/, int /*row*/, float/*sx*/, float/*sy*/, float fVal, 
                                                     const elemtype* samplePtr, float /*a00*/, float /*a10*/,
                                                     float/*denominator*/) {
        float gx = fVal * samplePtr[1], gy = fVal * samplePtr[2];
        return std::array<float, paramAmount>{gx, gy};
    }
};

template<> struct MotionTraits<MOTION_EUCLIDEAN> {
    enum { paramAmount = 3 };
    static inline void tail_handler_get_coord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col, 
                                              float numeratorX0,
                                              float numeratorY0,
                                              float /*denominator0*/,
                                              float a00,
                                              float a10,
                                              float /*a20*/)
    {
        denominator = 0;
        sx = (numeratorX0 + a00 * col);
        sy = (numeratorY0 + a10 * col);
    }

    template<typename elemtype> 
    static constexpr std::array<float, paramAmount> fillJacobian(int col, int row, float/*sx*/, float/*sy*/, float fVal, 
                                                     const elemtype* samplePtr, float a00, float a10,
                                                     float/*denominator*/) {
        float gx = fVal * samplePtr[1], gy = fVal * samplePtr[2];
        float hatX = -col * a10 - row * a00;
        float hatY = col * a00 - row * a10;
        float gz = gx * hatX + gy * hatY;
        return std::array<float, paramAmount>{gz, gx, gy};
    }
};

template<> struct MotionTraits<MOTION_AFFINE> {
    enum { paramAmount = 6};
    static inline void tail_handler_get_coord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col, 
                                              float numeratorX0,
                                              float numeratorY0,
                                              float /*denominator0*/,
                                              float a00,
                                              float a10,
                                              float /*a20*/)
    {
        denominator = 0;
        sx = (numeratorX0 + a00 * col);
        sy = (numeratorY0 + a10 * col);
    }

    template<typename elemtype> 
    static constexpr std::array<float, paramAmount> fillJacobian(int col, int row, float/*sx*/, float/*sy*/, float fVal, 
                                                     const elemtype* samplePtr, float /*a00*/, float /*a10*/,
                                                     float/*denominator*/) {
        float gx = fVal * samplePtr[1], gy = fVal * samplePtr[2];
        return std::array<float, paramAmount>{gx * col, gy * col, gx * row, gy * row, gx, gy};
    }
};

template<> struct MotionTraits<MOTION_HOMOGRAPHY> {
    enum { paramAmount = 8};
    static inline void tail_handler_get_coord(float& sx,
                                              float& sy,
                                              float& denominator,
                                              int col, 
                                              float numeratorX0,
                                              float numeratorY0,
                                              float denominator0,
                                              float a00,
                                              float a10,
                                              float a20)
    {
        denominator = 1.f / (col * a20 + denominator0);
        sx = (numeratorX0 + a00 * col) * denominator;
        sy = (numeratorY0 + a10 * col) * denominator;
    }
    
    template<typename elemtype> 
    static constexpr std::array<float, paramAmount> fillJacobian(int col, int row, float sx, float sy, float fVal, 
                                                     const elemtype* samplePtr, float/*a00*/, float/*a10*/,
                                                     float denominator) {
        float gx = fVal * float(samplePtr[1]) * denominator;
        float gy = fVal * float(samplePtr[2]) * denominator;
        float gz = -(gx * sx + gy * sy);
        return std::array<float, paramAmount>{gx * col, gy * col, gz * col, gx * row, gy * row, gz * row, gx, gy};
    }
};

inline void reinterpret(Mat& mat, int newdepth) { 
    mat.flags = (mat.flags & ~CV_MAT_DEPTH_MASK) | newdepth;
}

template<int N, class F>
class constexprForClass
{
public: 
    static inline void execute(F&& fVal) {
        constexprForClass<N-1, F>::execute(std::forward<F>(fVal));
        fVal(N-1);
    }
};

template<class F>
class constexprForClass<0, F>
{
public: 
    static inline void execute(F&&) {}
};

template<int N, class F>
constexpr void constexpr_for(F&& fVal) {
    constexprForClass<N, F>::execute(std::forward<F>(fVal));
}
template<int R, int C, class F>
class constexprForUpperTriangleClassOneRow
{
public: 
    static inline void execute(F&& fVal) {
        constexprForUpperTriangleClassOneRow<R, C-1, F>::execute(std::forward<F>(fVal));
        fVal(R, R + C - 1);
    }
};

template<int R, class F>
class constexprForUpperTriangleClassOneRow<R, 0, F>
{
public: 
    static inline void execute(F&&) {}
};

template<int R, int D, class F>
class constexprForUpperTriangleClass
{
public: 
    static inline void execute(F&& fVal) {
        constexprForUpperTriangleClass<R-1, D, F>::execute(std::forward<F>(fVal));
        constexprForUpperTriangleClassOneRow<R-1, D-R+1, F>::execute(std::forward<F>(fVal));
    }
};

template<int D, class F>
class constexprForUpperTriangleClass<0, D, F>
{
public: 
    static inline void execute(F&&) {}
};

template<int M, class F>
constexpr void constexpr_for_upper_triangle(F&& fVal) {
    constexprForUpperTriangleClass<M,M,F>::execute(std::forward<F>(fVal));
}

template<int MotionType>
constexpr int hessian_row_start(int row) {
    return row == 0 ? 0 : (MotionTraits<MotionType>::paramAmount - row + 1 + hessian_row_start<MotionType>(row - 1));
}

template<int motionType, typename elemtype>
static double image_hessian_proj_ECC(const Mat& map,
                           const Mat& sampleWithGrad,
                           const Mat& ref,
                           double& sampSum,
                           double& sampSqSum,
                           double& refSum,
                           double& refSqSum,
                           int& nz,
                           Mat& hessian,
                           Mat& sampleProj,
                           Mat& refProj,
                           int deltaY) {
    static_assert(std::is_same<float, elemtype>::value);
    constexpr int NPARAMS = MotionTraits<motionType>::paramAmount;

    CV_Assert(map.type() == CV_32F);
    CV_Assert(hessian.type() == CV_32F && sampleProj.type() == CV_32F && refProj.type() == CV_32F);
    if (sampleProj.size() != Size(1, NPARAMS) || refProj.size() != Size(1, NPARAMS)) {
        CV_Error(Error::BadImageSize, format("imageHessianProjECC: Wrong sample projection/reference projection size. 1x%d expected", NPARAMS));
    }
    if (hessian.size() != Size(NPARAMS, NPARAMS)) {
        CV_Error(Error::BadImageSize, format("imageHessianProjECC: Wrong hessian size. %dx%d expected", NPARAMS, NPARAMS));
    }
    if (!map.isContinuous()) {
        CV_Error(Error::BadStep, "imageHessianProjECC: Map should be continuous");
    }
    if (std::is_same<float, elemtype>::value) {
        CV_Assert(sampleWithGrad.type() == CV_32FC4 && ref.type() == CV_32FC2);
    }

    int hr = ref.rows;
    int wr = ref.cols;
    int hs = sampleWithGrad.rows;
    int ws = sampleWithGrad.cols;

    hessian = Mat::zeros(hessian.size(), hessian.type());
    sampleProj = Mat::zeros(sampleProj.size(), sampleProj.type());
    refProj = Mat::zeros(refProj.size(), refProj.type());

    const int MAX_STRIPES = 8;
    int stripesAmount = std::min(MAX_STRIPES, hr / deltaY);
    std::vector<std::vector<double>> hessPs(stripesAmount, std::vector<double>(NPARAMS * NPARAMS, 0.));
    std::vector<std::vector<double>> iprojs(stripesAmount, std::vector<double>(NPARAMS, 0.));
    std::vector<std::vector<double>> tprojs(stripesAmount, std::vector<double>(NPARAMS, 0.));
    std::vector<std::vector<double>> projSubs(stripesAmount, std::vector<double>(NPARAMS, 0.));
    std::vector<double> correlations(stripesAmount, 0.);

    // There is sophisticated story with masked and unmasked sums. We don't understand it for well,
    // but for some reason, in case, we are using masks algorithm becomes unstable if we apply masks
    // on images for calculationg all sums. Working solution is to use masks for hessians, projections,
    // correlations and not to use masks for simple statistics, like mean and standart deviation. In
    // the same time we need to calculate masked versions of mean and number of used pixels to
    // calculate correlation. So, we need pairs, like sampSums and sampMaskedSums or nzs and nzsMasked.

    std::vector<double> sampSums(stripesAmount, 0);
    std::vector<double> sampSqSums(stripesAmount, 0);
    std::vector<double> refSums(stripesAmount, 0);
    std::vector<double> refSqSums(stripesAmount, 0);
    std::vector<int> nzs(stripesAmount, 0);
    std::vector<double> sampMaskedSums(stripesAmount, 0);
    std::vector<double> refMaskedSums(stripesAmount, 0);
    std::vector<int> nzsMasked(stripesAmount, 0);

    double a00 = map.at<float>(0, 0);
    double a01 = map.at<float>(0, 1);
    double a02 = map.at<float>(0, 2);
    double a10 = map.at<float>(1, 0);
    double a11 = map.at<float>(1, 1);
    double a12 = map.at<float>(1, 2);
    double a20 = 0;
    double a21 = 0;
    double a22 = 0;
    if (motionType == MOTION_HOMOGRAPHY) {
        a20 = map.at<float>(2, 0);
        a21 = map.at<float>(2, 1);
        a22 = map.at<float>(2, 2);
    }

    const elemtype* samplePtr0 = sampleWithGrad.ptr<elemtype>(0);

    parallel_for_(Range(0, stripesAmount), [&](const Range& range) {
        int stripeNum = range.start;
        int ystart = (hr * stripeNum) / stripesAmount;
        ystart = roundUp(ystart, deltaY);
        int yend = (hr * (range.end)) / stripesAmount;
        // we don't store intermediate jacobian; instead, we iteratively update Hessian, sampleProj and refProj
        for (int y = ystart; y < yend; y += deltaY) {
            const elemtype* refPtr = ref.ptr<elemtype>(y);

            std::array<float, (NPARAMS * NPARAMS + NPARAMS) / 2> hessPcache{};
            std::array<float, NPARAMS> iprojCache{};
            std::array<float, NPARAMS> tprojCache{};
            std::array<float, NPARAMS> projSubCache{};

            const float numeratorX0 = y * a01 + a02;
            const float numeratorY0 = y * a11 + a12;
            const float denominator0 = y * a21 + a22;
            int x = 0;
            for (; x < wr; x++) { //Tail handler
                float sx, sy, denominator;
                MotionTraits<motionType>::tail_handler_get_coord(sx, sy, denominator, x, numeratorX0, numeratorY0,
                                                denominator0, a00, a10, a20);
                unsigned int ix = saturate_cast<unsigned>(sx);
                unsigned int iy = saturate_cast<unsigned>(sy);
                if ((static_cast<int>(ix < (unsigned int)ws) & static_cast<int>(iy < (unsigned int)hs)) != 0) {
                    const elemtype* samplePtr = samplePtr0 + iy * (ws * 4) + ix * 4;
                    float sampleVal = samplePtr[0];
                    float refVal = refPtr[2 * x];
                    sampSums[stripeNum] += sampleVal;
                    sampSqSums[stripeNum] += sampleVal * sampleVal;
                    refSums[stripeNum] += refVal;
                    refSqSums[stripeNum] += refVal * refVal;
                    nzs[stripeNum]++;
                    float fVal = float(samplePtr[3]);
                    float fValRef = float(refPtr[2 * x + 1]);
                    fVal = fVal == 0.f ? 0.f : 1.f;
                    fValRef = fValRef == 0.f ? 0.f : 1.f;
                    fVal *= fValRef;
                    sampleVal *= fVal;
                    refVal *= fVal;
                    nzsMasked[stripeNum] += fVal;
                    sampMaskedSums[stripeNum] += sampleVal;
                    refMaskedSums[stripeNum] += refVal;
                    std::array<float, NPARAMS> jac = MotionTraits<motionType>::fillJacobian(x, y, sx, sy, 
                                                                                            fVal, samplePtr, a00,
                                                                                            a10, denominator);
                    constexpr_for_upper_triangle<NPARAMS>([&](int row_i, int col_i) {
                        hessPcache[hessian_row_start<motionType>(row_i) + (col_i - row_i)] += jac[row_i] * jac[col_i];
                    });
                    constexpr_for<NPARAMS>([&](int elem) {
                        iprojCache[elem] += jac[elem] * sampleVal;
                        tprojCache[elem] += jac[elem] * refVal;
                        projSubCache[elem] += jac[elem] * fVal;
                    });
                    correlations[stripeNum] += sampleVal * refVal;
                }
            }

            constexpr_for_upper_triangle<NPARAMS>([&](int row, int col) {
                hessPs[stripeNum][row * NPARAMS + col] += hessPcache[hessian_row_start<motionType>(row) + (col - row)];
            });
            constexpr_for<NPARAMS>([&](int elem) {
                iprojs[stripeNum][elem] += iprojCache[elem];
                tprojs[stripeNum][elem] += tprojCache[elem];
                projSubs[stripeNum][elem] += projSubCache[elem];
            });
        }
    });
    std::vector<double> hessP(NPARAMS * NPARAMS, 0.);
    std::vector<double> iproj(NPARAMS, 0.);
    std::vector<double> tproj(NPARAMS, 0.);
    double sampMaskedSum = 0;
    double refMaskedSum = 0;
    double correlation = 0;
    sampSum = sampSqSum = refSum = refSqSum = nz = 0;
    int nzMasked = 0;

    for (int stripeNum = 0; stripeNum < stripesAmount; stripeNum++) {
        correlation += correlations[stripeNum];
        sampSum += sampSums[stripeNum];
        sampSqSum += sampSqSums[stripeNum];
        refSum += refSums[stripeNum];
        refSqSum += refSqSums[stripeNum];
        sampMaskedSum += sampMaskedSums[stripeNum];
        refMaskedSum += refMaskedSums[stripeNum];
        nz += nzs[stripeNum];
        nzMasked += nzsMasked[stripeNum];
    }
    double scale = nz == 0 ? 0. : 1. / nz;
    double sampMean = sampSum * scale;
    double refMean = refSum * scale;
    correlation += nzMasked * sampMean * refMean - sampMaskedSum * refMean - refMaskedSum * sampMean;

    for (int stripeNum = 0; stripeNum < stripesAmount; stripeNum++) {
        for (int hessNum = 0; hessNum < static_cast<int>(hessP.size()); hessNum++) {
            hessP[hessNum] += hessPs[stripeNum][hessNum];
        }
        for (int projNum = 0; projNum < NPARAMS; projNum++) {
            iproj[projNum] += iprojs[stripeNum][projNum] - projSubs[stripeNum][projNum] * sampMean;
            tproj[projNum] += tprojs[stripeNum][projNum] - projSubs[stripeNum][projNum] * refMean;
        }
    }

    constexpr_for_upper_triangle<NPARAMS>([&](int row, int col) {
        hessP[col * NPARAMS + row] = hessP[row * NPARAMS + col];
    });

    Mat(NPARAMS, NPARAMS, CV_64F, hessP.data()).convertTo(hessian, CV_32F);
    Mat(NPARAMS, 1, CV_64F, iproj.data()).convertTo(sampleProj, CV_32F);
    Mat(NPARAMS, 1, CV_64F, tproj.data()).convertTo(refProj, CV_32F);

    return correlation;
}

static void optimize_ECC(Mat& sampleWithGrad,
                 const Mat& reference,
                 Mat& map,
                 int motionType,
                 double* rho,
                 double* lastRho,
                 int deltaY,
                 int nparams) {
    // warp-back portion of the inputImage and gradients to the coordinate space of the referenceFloat
    double correlation = 0;

    // matrices needed for solving linear equation system for maximizing ECC
    Mat hessian = Mat(nparams, nparams, CV_32F);
    Mat hessianInv = Mat(nparams, nparams, CV_32F);
    Mat sampleProjection = Mat(nparams, 1, CV_32F);
    Mat referenceProjection = Mat(nparams, 1, CV_32F);
    Mat sampleProjectionHessian = Mat(nparams, 1, CV_32F);
    Mat errorProjection = Mat(nparams, 1, CV_32F);
    Mat deltaP = Mat(nparams, 1, CV_32F);

    double sampSum;
    double sampSqSum;
    double referenceSum;
    double referenceSqSum;
    int nz;

    {  // if(imageWithGrad.type() == CV_32FC4)
        if (motionType == MOTION_TRANSLATION) {
            correlation = image_hessian_proj_ECC<MOTION_TRANSLATION, float>(map,
                                                                       sampleWithGrad,
                                                                       reference,
                                                                       sampSum,
                                                                       sampSqSum,
                                                                       referenceSum,
                                                                       referenceSqSum,
                                                                       nz,
                                                                       hessian,
                                                                       sampleProjection,
                                                                       referenceProjection,
                                                                       deltaY);
        } else if (motionType == MOTION_EUCLIDEAN) {
            correlation = image_hessian_proj_ECC<MOTION_EUCLIDEAN, float>(map,
                                                                       sampleWithGrad,
                                                                       reference,
                                                                       sampSum,
                                                                       sampSqSum,
                                                                       referenceSum,
                                                                       referenceSqSum,
                                                                       nz,
                                                                       hessian,
                                                                       sampleProjection,
                                                                       referenceProjection,
                                                                       deltaY);
        } else if (motionType == MOTION_AFFINE) {
            correlation = image_hessian_proj_ECC<MOTION_AFFINE, float>(map,
                                                                       sampleWithGrad,
                                                                       reference,
                                                                       sampSum,
                                                                       sampSqSum,
                                                                       referenceSum,
                                                                       referenceSqSum,
                                                                       nz,
                                                                       hessian,
                                                                       sampleProjection,
                                                                       referenceProjection,
                                                                       deltaY);
        } else {
            correlation = image_hessian_proj_ECC<MOTION_HOMOGRAPHY, float>(map,
                                                                        sampleWithGrad,
                                                                        reference,
                                                                        sampSum,
                                                                        sampSqSum,
                                                                        referenceSum,
                                                                        referenceSqSum,
                                                                        nz,
                                                                        hessian,
                                                                        sampleProjection,
                                                                        referenceProjection,
                                                                        deltaY);
        }
    }
    double scale = nz == 0 ? 0. : 1. / nz;
    double sampMean = sampSum * scale;
    double refMean = referenceSum * scale;
    double sampStd = std::sqrt(std::max(sampSqSum * scale - sampMean * sampMean, 0.));
    double refStd = std::sqrt(std::max(referenceSqSum * scale - refMean * refMean, 0.));

    // inverse of Hessian
    hessianInv = hessian.inv();
    // calculate enhanced correlation coefficient (ECC)->rho
    *lastRho = *rho;
    double refNorm = std::sqrt(nz * refStd * refStd);
    double sampNorm = std::sqrt(nz * sampStd * sampStd);

    *rho = correlation / (sampNorm * refNorm);
    if ((bool)cvIsNaN(*rho)) {
        CV_Error(Error::StsNoConv, "NaN encountered.");
    }

    // calculate the parameter lambda to account for illumination variation
    sampleProjectionHessian = hessianInv * sampleProjection;
    const double lambdaN = (sampNorm * sampNorm) - sampleProjection.dot(sampleProjectionHessian);
    const double lambdaD = correlation - referenceProjection.dot(sampleProjectionHessian);

    if (lambdaD <= 0.0) {
        CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. "
            "Images may be uncorrelated or non-overlapped");
    }
    const double lambda = (lambdaN / lambdaD);

    // estimate the update step delta_p
    errorProjection = lambda * referenceProjection - sampleProjection;
    gemm(hessianInv, errorProjection, 1., noArray(), 0., deltaP);

    // update warping matrix
    update_warping_matrix_ECC(map, deltaP, motionType);
}

static Mat prepare_gradients(const Mat& sample) {
    CV_Assert(sample.type() == CV_32FC2 || sample.type() == CV_16FC2);

    const int ws = sample.cols;
    const int hs = sample.rows;

    Mat sampleWithGrad;
    int ntasks = std::min(4, hs);

    {
        sampleWithGrad = Mat(hs, ws, CV_32FC4);
        float* dstPtr = sampleWithGrad.ptr<float>();
        parallel_for_(Range(0, ntasks), [&](const Range& range) {
            int rowstart = range.start * hs / ntasks;
            int rowend = range.end * hs / ntasks;
            for (int row = rowstart; row < rowend; row++) {
                const float* sampleCurLine = sample.ptr<float>(row);
                const float* samplePrevLine = sample.ptr<float>(std::max(row - 1, 0));
                const float* sampleNextLine = sample.ptr<float>(std::min(row + 1, hs - 1));
                float gradDivY = (row > 0 && row + 1 < hs) ? 0.5 : 0.25;
                int col = 0;
                for (; col < ws; col++) {
                    int prevCol = std::max(col - 1, 0);
                    int nextCol = std::min(col + 1, ws - 1);
                    float gradDivX = (col > 0 && col + 1 < ws) ? 0.5 : 0.25;
                    dstPtr[row * ws * 4 + col * 4] = sampleCurLine[2 * col];
                    dstPtr[row * ws * 4 + col * 4 + 1] =
                        gradDivX * (sampleCurLine[2 * nextCol] - sampleCurLine[2 * prevCol]);
                    dstPtr[row * ws * 4 + col * 4 + 2] = gradDivY * (sampleNextLine[2 * col] - samplePrevLine[2 * col]);
                    dstPtr[row * ws * 4 + col * 4 + 3] = sampleCurLine[2 * col + 1];
                }
            }
        }, ntasks);
    }
    return sampleWithGrad;
}

static void build_pyramid(InputArray inputImage,
                  MatPyramid& imgPyramid,
                  InputArray& mask,
                  MatPyramid& maskPyramid,
                  int numberOfPyramidsLevel) {
    imgPyramid.resize(numberOfPyramidsLevel);
    inputImage.getMat().convertTo(imgPyramid[0], CV_8UC1);
    maskPyramid.resize(numberOfPyramidsLevel);
    if (!mask.empty()) {
        mask.getMat().convertTo(maskPyramid[0], CV_8UC1);
    }
    for (int pyrLevel = 0; pyrLevel < numberOfPyramidsLevel - 1; ++pyrLevel) {
        Size size = Size((imgPyramid[pyrLevel].cols + 1) / 2, (imgPyramid[pyrLevel].rows + 1) / 2);
        pyrDown(imgPyramid[pyrLevel], imgPyramid[pyrLevel + 1], size);
        if (!mask.empty()) {
            pyrDown(maskPyramid[pyrLevel], maskPyramid[pyrLevel + 1], size);
            threshold(maskPyramid[pyrLevel + 1], maskPyramid[pyrLevel + 1], 254, 0xff, THRESH_BINARY);
        }
    }
}

static Mat splice_with_mask(const Mat& image, const Mat& mask) {
    CV_Assert(image.type() == CV_32F && (mask.empty() || mask.type() == CV_8U));
    if (!mask.empty() && image.size() != mask.size()) {
        CV_Error(Error::BadImageSize, "spliceWithMask: Mask and image have to be of same size.");
    }
    const int hs = image.rows;
    const int ws = image.cols;

    Mat result;
    int ntasks = std::min(4, hs);
    {
        union conv_ {
            uint32_t valU;
            float val;
            conv_() : valU(0xffffffff) {}
        } conv;
        result = Mat(hs, ws, CV_32FC2);
        parallel_for_(Range(0, ntasks), [&](const Range& range) {
            int rowstart = range.start * hs / ntasks;
            int rowend = range.end * hs / ntasks;
            for (int row = rowstart; row < rowend; row++) {
                float* dstPtr = result.ptr<float>(row);
                const float* srcPtr = image.ptr<float>(row);
                const uint8_t* maskPtr = !mask.empty() ? mask.ptr<uint8_t>(row) : nullptr;
                int col = 0;
                for (; col < ws; col++) {
                    dstPtr[col * 2] = srcPtr[col];
                    dstPtr[col * 2 + 1] = (!maskPtr || maskPtr[col]) ? conv.val : 0;
                }
            }
        }, ntasks);
    }
    return result;
}

static void scale_warp_matrix(Mat& warpMatrix, float scale) {
    if (warpMatrix.rows == 3) {
        Mat invertScaleMat = Mat(3, 3, CV_32F, 0.f);
        invertScaleMat.at<float>(0, 0) = 1.f / scale;
        invertScaleMat.at<float>(1, 1) = 1.f / scale;
        invertScaleMat.at<float>(2, 2) = 1.f;
        Mat scaleMatrix = invertScaleMat.clone();
        scaleMatrix.at<float>(0, 0) = scale;
        scaleMatrix.at<float>(1, 1) = scale;
        gemm(warpMatrix, invertScaleMat, 1., noArray(), 0., warpMatrix);
        gemm(scaleMatrix, warpMatrix, 1., noArray(), 0., warpMatrix);
        // Normalization, internal algorithms assumes, that a22 = 1.0f
        for (int mel = 0; mel < 8; mel++) {
            (reinterpret_cast<float*>(warpMatrix.data))[mel] /=
                (reinterpret_cast<float*>(warpMatrix.data))[8];
        }
        (reinterpret_cast<float*>(warpMatrix.data))[8] = 1.f;
    } else {
        warpMatrix.at<float>(0, 2) *= scale;
        warpMatrix.at<float>(1, 2) *= scale;
    }
}

static void check_params(const MatPyramid& referencePyramid,
                 const MatPyramid& samplePyramid,
                 Mat& map,
                 int motionType,
                 TermCriteria criteria,
                 std::vector<int>& itersPerLevel,
                 int numberOfPyramidsLevel) {
    if (itersPerLevel.empty()) {
        itersPerLevel.resize(numberOfPyramidsLevel, criteria.maxCount);
    }
    CV_Assert(static_cast<int>(itersPerLevel.size()) == numberOfPyramidsLevel);
    CV_Assert(!referencePyramid.empty());
    for (const auto& lvl : referencePyramid) {
        CV_Assert(!lvl.empty() && lvl.type() == referencePyramid[0].type());
    }
    CV_Assert(!samplePyramid.empty());
    for (const auto& lvl : samplePyramid) {
        CV_Assert(!lvl.empty() && lvl.type() == samplePyramid[0].type());
    }
    CV_Assert(samplePyramid.size() == referencePyramid.size() && samplePyramid.size() == itersPerLevel.size());

    // If the user passed an un-initialized warpMatrix, initialize to identity
    if (referencePyramid[0].type() != CV_32FC2 && referencePyramid[0].type() != CV_16FC2) {
        CV_Error(Error::StsError, "Reference pyramid have to be prepared via prepareReferencePyramid function");
    }
    // accept only 1-channel images
    CV_Assert(samplePyramid[0].type() == CV_32FC2 || samplePyramid[0].type() != CV_16FC2);
    CV_Assert(map.type() == CV_32FC1);
    if (map.cols != 3 || (map.rows != 2 && map.rows != 3)) {
        CV_Error(Error::BadImageSize, "warpMatrix has incorrect size");
    }

    if (motionType != MOTION_TRANSLATION && motionType != MOTION_EUCLIDEAN && 
        motionType != MOTION_AFFINE && motionType != MOTION_HOMOGRAPHY) {
        CV_Error(Error::StsError, "Incorrect motion type");
    }

    if (motionType == MOTION_HOMOGRAPHY && map.rows != 3) {
        CV_Error(Error::BadImageSize, "warpMatrix has incorrect size");
    }

    if (!((bool)(criteria.type & TermCriteria::COUNT) || (bool)(criteria.type & TermCriteria::EPS))) {
        CV_Error(Error::StsError, "Incorrect stop criteria");
    }
}

MatPyramid cv::prepareECCPyramid(InputArray image,
                             InputArray imageMask,  // Can be empty
                             int gaussFiltSize,
                             int numberOfPyramidsLevel) {
    MatPyramid imagePyramid, maskPyramid;
    build_pyramid(image, imagePyramid, imageMask, maskPyramid, numberOfPyramidsLevel);
    for (int lvl = 0; lvl < numberOfPyramidsLevel; lvl++) {
        Mat imgFloat;
        imagePyramid[lvl].convertTo(imgFloat, CV_32F, 1. / 255.);
        if (gaussFiltSize != 0) {
            GaussianBlur(imgFloat, imgFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        }
        imagePyramid[lvl] = splice_with_mask(
            imgFloat,
            (static_cast<int>(maskPyramid.size()) > lvl && !maskPyramid[lvl].empty()) ? maskPyramid[lvl] : Mat());
    }
    return imagePyramid;
}

double cv::findTransformECCPyr(InputArray reference,
                        InputArray sample,
                        InputOutputArray warpMatrix,
                        const int motionType,
                        const TermCriteria criteria,
                        const std::vector<int>& itersPerLevel,
                        InputArray referenceMask,
                        InputArray sampleMask,
                        const int gaussFiltSize,
                        const int numberOfPyramidsLevel) {
    MatPyramid referencePyramid =
        prepareECCPyramid(reference, referenceMask, gaussFiltSize, numberOfPyramidsLevel);
    return findTransformECCPyr(referencePyramid,
                            sample,
                            warpMatrix,
                            motionType,
                            criteria,
                            itersPerLevel,
                            sampleMask,
                            gaussFiltSize,
                            numberOfPyramidsLevel);
}

double cv::findTransformECCPyr(const MatPyramid& referencePyramid,
                        InputArray sample,
                        InputOutputArray warpMatrix,
                        const int motionType,
                        const TermCriteria criteria,
                        const std::vector<int>& itersPerLevel,
                        InputArray sampleMask,
                        const int gaussFiltSize,
                        const int numberOfPyramidsLevel) {
    MatPyramid samplePyramid = prepareECCPyramid(sample, sampleMask, gaussFiltSize, numberOfPyramidsLevel);
    return findTransformECCPyr(referencePyramid,
                            samplePyramid,
                            warpMatrix,
                            motionType,
                            criteria,
                            itersPerLevel,
                            gaussFiltSize,
                            numberOfPyramidsLevel);
}

double cv::findTransformECCPyr(InputArray reference,
                        const MatPyramid& samplePyramid,
                        InputOutputArray warpMatrix,
                        const int motionType,
                        const TermCriteria criteria,
                        const std::vector<int>& itersPerLevel,
                        InputArray referenceMask,
                        const int gaussFiltSize,
                        const int numberOfPyramidsLevel) {
    MatPyramid referencePyramid =
        prepareECCPyramid(reference, referenceMask, gaussFiltSize, numberOfPyramidsLevel);
    return findTransformECCPyr(referencePyramid,
                            samplePyramid,
                            warpMatrix,
                            motionType,
                            criteria,
                            itersPerLevel,
                            gaussFiltSize,
                            numberOfPyramidsLevel);
}

double cv::findTransformECCPyr(const MatPyramid& referencePyramid,
                        const MatPyramid& samplePyramid,
                        InputOutputArray warpMatrixA,
                        const int motionType,
                        const TermCriteria criteria,
                        const std::vector<int>& itersPerLevel,
                        const int /*gaussFiltSize*/,
                        const int numberOfPyramidsLevel) {
    Mat& warpMatrix = warpMatrixA.getMatRef();
    std::vector<int> itersPerLevelCopy = itersPerLevel;
    // If the user passed an un-initialized warpMatrix, initialize to identity
    if (warpMatrix.empty()) {
        int rowCount = motionType == MOTION_HOMOGRAPHY ? 3 : 2;
        warpMatrix = Mat::eye(rowCount, 3, CV_32FC1);
    }

    check_params(referencePyramid,
                samplePyramid,
                warpMatrix,
                motionType,
                criteria,
                itersPerLevelCopy,
                numberOfPyramidsLevel);

    int nparams = MotionTraits<MOTION_AFFINE>::paramAmount; // default
    switch (motionType) {
        case MOTION_TRANSLATION: 
            nparams = MotionTraits<MOTION_TRANSLATION>::paramAmount;
            break;
        case MOTION_EUCLIDEAN:
            nparams = MotionTraits<MOTION_EUCLIDEAN>::paramAmount;
            break;
        case MOTION_HOMOGRAPHY:
            nparams = MotionTraits<MOTION_HOMOGRAPHY>::paramAmount;
            break;
    }

    const std::vector<int> numberOfIterations = ((criteria.type & TermCriteria::COUNT) != 0)
                                                    ? itersPerLevelCopy
                                                    : std::vector<int>(numberOfPyramidsLevel, 200);
    const double terminationEPS = (bool)(criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

    // Scale warp matrix multiple times to lower pyramid level
    for (int pyrLevel = 0; pyrLevel < numberOfPyramidsLevel - 1; pyrLevel++) {
        scale_warp_matrix(warpMatrix, 0.5);
    }
    double rho = -1;
    for (int pyrLevel = numberOfPyramidsLevel - 1; pyrLevel >= 0; --pyrLevel) {
        const int hr = referencePyramid[pyrLevel].rows;

        Mat sampleWithGrad = prepare_gradients(samplePyramid[pyrLevel]);

        const int LOW_SIZE = 200;
        int deltaY = hr < LOW_SIZE ? 1 : 2;

        // iteratively update mapMatrix
        double lastRho = -terminationEPS;
        for (int i = 1; (i <= numberOfIterations[pyrLevel]) && (fabs(rho - lastRho) >= terminationEPS); i++) {
            optimize_ECC(sampleWithGrad, referencePyramid[pyrLevel], warpMatrix, motionType, &rho, &lastRho, deltaY, nparams);
        }
        if (pyrLevel > 0) {
            scale_warp_matrix(warpMatrix, 2);
        }
    }
    // return final correlation coefficient
    return rho;
}
/* End of file. */