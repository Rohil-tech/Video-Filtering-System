/**
 * Rohil Kulshreshtha
 * January 18, 2026
 * CS 5330 - PR-CV - Assignment 1
 * 
 * Header file for image filter functions.
*/


#ifndef FILTER_H
#define FILTER_H


// filter prototypes
int opencvGreyScale(cv::Mat &src, cv::Mat &dst);
int customGreyScale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int vignette(cv::Mat &src, cv::Mat &dst, float intensity);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int warmth(cv::Mat &src, cv::Mat &dst, float intensity);
int coolness(cv::Mat &src, cv::Mat &dst, float intensity);
int exposure(cv::Mat &src, cv::Mat &dst, float exposure);
int mirror(cv::Mat &src, cv::Mat &dst);
int depthBlur(cv::Mat &src, cv::Mat &dst);
int depthColorGrade(cv::Mat &src, cv::Mat &dst);
int depthVisualizationHot(cv::Mat &src, cv::Mat &dst);
int depthVisualizationCool(cv::Mat &src, cv::Mat &dst);
int emboss(cv::Mat &src, cv::Mat &dst);
int negative(cv::Mat &src, cv::Mat &dst);
int sparkleHalo(cv::Mat &src, cv::Mat &dst);
int isolateRed(cv::Mat &src, cv::Mat &dst);
int isolateBlue(cv::Mat &src, cv::Mat &dst);
int isolateGreen(cv::Mat &src, cv::Mat &dst);
int isolateYellow(cv::Mat &src, cv::Mat &dst);


#endif