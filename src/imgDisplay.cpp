/**
 * Rohil Kulshreshtha
 * January 17, 2026
 * CS 5330 - PR-CV - Assignment 1
 * 
 * Display an image in a window and handle user keypresses.
*/

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include "opencv2/opencv.hpp"
#include "filterApp.h"

/** 
 * Main function to display an image and handle user input.
*/
int main(int argc, char *argv[]) {
    
    // Check if the user provided a filename
    if( argc < 2 ) {
        printf("Usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }
    
    // Setting the OpenCV logging to only display warning and error messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    cv::Mat image;
    char filename[256];
    
    // Store the filename from command line argument
    strcpy(filename, argv[1]);
    
    // Read the image from file
    image = cv::imread(filename);
    
    // Verify the image was successfully loaded
    if (image.data == NULL) {
        printf("ERROR: Unable to read image file '%s'\n", filename);
        exit(-2);
    }
    
    // Display the image in the window
    const char* windowName = "Image Display";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    printf("\n=== Image Information ===\n");
    printf("Filename:  %s\n", filename);
    printf("Size:      %d x %d pixels\n", image.cols, image.rows);
    printf("Channels:  %d\n", image.channels());
    printf("Depth:     %d bits per channel\n", (int)(image.elemSize() / image.channels()) * 8);
    
    // Lambda captures 'image' by reference - no global needed!
    runFilterApp([&image]() { 
        return image;
    }, windowName, image.size(), false);
    
    // Clean up - destroy the image window
    cv::destroyWindow(windowName);
    
    printf("Application terminated successfully.\n");
    return 0;
}