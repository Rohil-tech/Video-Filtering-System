/**
 * Rohil Kulshreshtha
 * January 20, 2026
 * CS 5330 - PR-CV - Assignment 1
 * 
 * Header file for shared filter application logic which contains the main filter UI and processing loop.
*/

#ifndef FILTERAPP_H
#define FILTERAPP_H

/**
 * Runs the interactive filter application loop.
 * Handles keyboard input and applies filters to frames.
 * Works for both static images and video streams.
 */
int runFilterApp(std::function<cv::Mat()> getFrame, const char* windowName, cv::Size refS, bool isCameraMode);

#endif