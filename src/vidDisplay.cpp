/**
 * Rohil Kulshreshtha
 * January 18, 2026
 * CS 5330 - PR-CV - Assignment 1
 * 
 * Capture and display live video from a camera.
*/

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include "opencv2/opencv.hpp"
#include "filterApp.h"

/**
 * Main function to capture and display live video from camera.
*/
int main(int argc, char *argv[]) {
    
    // Setting the OpenCV logging to only display warning and error messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    // Initialize and open the video device
    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);
    if(!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }
    
    printf("=== Live Camera Mode ===\n");
    printf("Camera opened successfully\n");

    // Get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH), (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    // Create a named window for video display
    const char* windowName = "Video";
    cv::namedWindow(windowName, 1);
    
    cv::Mat frame;

    runFilterApp([&capdev]() { 
        cv::Mat frame;
        *capdev >> frame;
        return frame;
    }, windowName, refS, true);

    // Clean up - destroy window and release camera
    cv::destroyWindow(windowName);
    delete capdev;

    printf("Application terminated successfully.\n");
    return 0;
}