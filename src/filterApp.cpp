/**
 * Rohil Kulshreshtha
 * January 20, 2026
 * CS 5330 - Assignment 1
 * 
 * Implementation of shared filter application logic.
 * Contains the main filter UI and processing loop used by both imgDisplay and vidDisplay programs.
*/

#include <cstdio>
#include <chrono>
#include <string>
#include "opencv2/opencv.hpp"
#include "filter.h"
#include "faceDetect.h"
#include "filterApp.h"

/**
 * Runs the interactive filter application loop.
 * This function contains all the filter logic and keyboard handling
 * shared between image and video display modes.
*/
int runFilterApp(std::function<cv::Mat()> getFrame, const char* windowName, cv::Size refS, bool isCameraMode) {

    // Initializing video recording state variables
    bool isRecording = false;
    cv::VideoWriter videoWriter;
    int recordedFrames = 0;
    const int MAX_FRAMES = 900;
    std::chrono::steady_clock::time_point recordingStartTime;

    // Helper function to generate timestamped video filename
    auto generateVideoFilename = []() -> std::string {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        localtime_s(&tm_now, &t);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm_now);
        return std::string("recording_") + buffer + ".mp4";
    };

    // Helper function to initialize VideoWriter
    auto initializeVideoWriter = [&](const std::string &filename, cv::Size frameSize) -> bool {
        // Initializing the video writer to MP4
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoWriter.open(filename, fourcc, 30.0, frameSize);
        if (videoWriter.isOpened()) {
            printf("Using MP4 format (H.264)\n");
            return true;
        }
        printf("ERROR: Could not initialize video writer\n");
        return false;
    };

    // Helper function for countdown before recording
    auto showCountdown = [&](std::function<cv::Mat()> getProcessedFrame) {
        const int COUNTDOWN_START = 5;
        for (int count = COUNTDOWN_START; count >= 1; --count) {
            auto start = std::chrono::steady_clock::now();
            while (true) {
                cv::Mat frame = getProcessedFrame();
                if (frame.empty()) break;
                cv::Mat display = frame.clone();
                
                // Countdown text
                std::string text = std::to_string(count);
                int fontFace = cv::FONT_HERSHEY_DUPLEX;
                double fontScale = 4.0;
                int thickness = 8;
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
                cv::Point pos((display.cols - textSize.width) / 2, (display.rows + textSize.height) / 2);

                // White text
                cv::putText(display, text, pos, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
                cv::imshow(windowName, display);

                if (cv::waitKey(1) == 'q') {
                    return;
                }

                auto now = std::chrono::steady_clock::now();
                float elapsed = std::chrono::duration<float>(now - start).count();

                if (elapsed >= 1.0f) {
                    break;
                }
            }
        }

        printf("Recording started!\n");
    };

    
    // Defining filter modes
    enum FilterMode {
        FILTER_NONE = 0,
        FILTER_OPENCV_GREYSCALE,
        FILTER_CUSTOM_GREYSCALE,
        FILTER_SEPIA,
        FILTER_SOBEL_X,
        FILTER_SOBEL_Y,
        FILTER_GRADIENT_MAGNITUDE,
        FILTER_BLUR_QUANTIZE,
        FILTER_DEPTH_VIS_HOT,
        FILTER_DEPTH_VIS_COOL,
        FILTER_DEPTH_BLUR,
        FILTER_DEPTH_COLOR,
        FILTER_EMBOSS,
        FILTER_NEGATIVE,
        FILTER_COLOR_ISOLATION_RED,
        FILTER_COLOR_ISOLATION_BLUE,
        FILTER_COLOR_ISOLATION_GREEN,
        FILTER_COLOR_ISOLATION_YELLOW,
    };

    // Initializing filter state variables
    FilterMode activeFilter = FILTER_NONE;
    bool mirrorMode = false;
    bool vignetteMode = false;
    bool blurredMode = false;
    bool faceDetectMode = false;
    bool haloMode = false;
    float vignetteIntensity = 0.5f;
    float exposureValue = 1.0f;
    float warmthIntensity = 0.0f;
    float coolnessIntensity = 0.0f;

    auto printMenu = []() {
        // Printing instructions to console for user help
        printf("\n============ FILTER APPLICATION ============\n");
        printf("\n=== Primary Filters (Mutually Exclusive) ===\n");
        printf("Press 'g' - OpenCV Greyscale\n");
        printf("Press 'h' - Custom Greyscale\n");
        printf("Press 'p' - Sepia Tone\n");
        printf("Press 'x' - Sobel X (Vertical Edges)\n");
        printf("Press 'y' - Sobel Y (Horizontal Edges)\n");
        printf("Press 'm' - Gradient Magnitude\n");
        printf("Press 'l' - Blur Quantize (Cartoon)\n");
        printf("Press 'n' - Negative (Color Invert)\n");
        printf("Press 'z' - Emboss (3D Relief)\n");
        printf("Press 'd' - Depth Modes (cycle: hot/cool/bokeh/color-grading)\n");
        printf("Press 'o' - Color Modes (cycle: red/blue/green/yellow)\n");

        printf("\n=== Secondary Effects (Stackable) ===\n");
        printf("Press 'v' - Vignette (+/- to adjust intensity) (0.0-2.0)\n");
        printf("Press 'b' - Blur (5x5 Gaussian)\n");
        printf("Press 'w' - Warmth (cycle 0.0-2.0)\n");
        printf("Press 'e' - Coolness (cycle 0.0-2.0)\n");
        printf("Press 'f' - Face Detection\n");
        printf("Press 't' - Sparkle Halo (above faces)\n");
        printf("Press 'r' - Mirror/Flip\n");
        printf("Press '1' - Increase Brightness\n");
        printf("Press '2' - Decrease Brightness\n");

        printf("\n=== Actions ===\n");
        printf("Press 'c' - Clear All Filters\n");
        printf("Press 'i' - Display Info about active filters and effects\n");
        printf("Press 's' - Save Screenshot\n");
        printf("Press 'k' - Record Video (30s max, camera mode only)\n");
        printf("Press '~' - To see the menu again\n");
        printf("Press 'q' - Quit\n");
        printf("============================================\n\n");
    };

    printMenu();
    
    char key = 0;
    cv::Mat vignetteFrame, blurredFrame, warmFrame, coolFrame, exposedFrame, faceFrame, displayFrame, haloFrame, displayWithIndicator;

    auto getProcessedFrame = [&]() -> cv::Mat {
        cv::Mat frame = getFrame();
        if (frame.empty()) {
            return cv::Mat();
        }

        cv::Mat processedFrame;

        // --- PRIMARY FILTER ---
        switch(activeFilter) {
            case FILTER_OPENCV_GREYSCALE:
                opencvGreyScale(frame, processedFrame);
                break;
            case FILTER_CUSTOM_GREYSCALE:
                customGreyScale(frame, processedFrame);
                break;
            case FILTER_SEPIA:
                sepia(frame, processedFrame);
                break;
            case FILTER_SOBEL_X: {
                cv::Mat tmp;
                sobelX3x3(frame, tmp);
                cv::convertScaleAbs(tmp, processedFrame);
                break;
            }
            case FILTER_SOBEL_Y: {
                cv::Mat tmp;
                sobelY3x3(frame, tmp);
                cv::convertScaleAbs(tmp, processedFrame);
                break;
            }
            case FILTER_GRADIENT_MAGNITUDE: {
                cv::Mat sx, sy;
                sobelX3x3(frame, sx);
                sobelY3x3(frame, sy);
                magnitude(sx, sy, processedFrame);
                break;
            }
            case FILTER_BLUR_QUANTIZE:
                blurQuantize(frame, processedFrame, 10);
                break;
            case FILTER_DEPTH_VIS_HOT:
                depthVisualizationHot(frame, processedFrame);
                break;
            case FILTER_DEPTH_VIS_COOL:
                depthVisualizationCool(frame, processedFrame);
                break;
            case FILTER_DEPTH_BLUR:
                depthBlur(frame, processedFrame);
                break;
            case FILTER_DEPTH_COLOR:
                depthColorGrade(frame, processedFrame);
                break;
            case FILTER_EMBOSS:
                emboss(frame, processedFrame);
                break;
            case FILTER_NEGATIVE:
                negative(frame, processedFrame);
                break;
            case FILTER_COLOR_ISOLATION_RED:
                isolateRed(frame, processedFrame);
                break;
            case FILTER_COLOR_ISOLATION_BLUE:
                isolateBlue(frame, processedFrame);
                break;
            case FILTER_COLOR_ISOLATION_GREEN:
                isolateGreen(frame, processedFrame);
                break;
            case FILTER_COLOR_ISOLATION_YELLOW:
                isolateYellow(frame, processedFrame);
                break;
            default:
                processedFrame = frame;
                break;
        }

        // --- SECONDARY EFFECTS (same order as main loop) ---
        cv::Mat tmp1, tmp2;

        if (vignetteMode) {
            vignette(processedFrame, tmp1, vignetteIntensity);
        } else {
            tmp1 = processedFrame;
        }

        if (blurredMode) {
            blur5x5_2(tmp1, tmp2);
        } else {
            tmp2 = tmp1;
        }

        if (warmthIntensity > 0.0f) {
            warmth(tmp2, tmp1, warmthIntensity);
        } else {
            tmp1 = tmp2;
        }

        if (coolnessIntensity > 0.0f) {
            coolness(tmp1, tmp2, coolnessIntensity);
        } else {
            tmp2 = tmp1;
        }

        if (exposureValue != 1.0f) {
            exposure(tmp2, tmp1, exposureValue);
        } else {
            tmp1 = tmp2;
        }

        if (faceDetectMode) {
            cv::Mat gray;
            cv::cvtColor(tmp1, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            detectFaces(gray, faces);
            tmp1.copyTo(tmp2);
            drawBoxes(tmp2, faces);
        } else {
            tmp2 = tmp1;
        }

        if (haloMode) {
            sparkleHalo(tmp2, tmp1);
        } else {
            tmp1 = tmp2;
        }

        if (mirrorMode) {
            mirror(tmp1, tmp2);
        } else {
            tmp2 = tmp1;
        }

        return tmp2;
    };


    // Main processing loop
    while(key != 'q') {

        // Get the next frame (either static image or video frame)
        cv::Mat frame = getFrame();
        
        // Check if frame is empty
        if (frame.empty()) {
            printf("ERROR: Frame is empty\n");
            break;
        }

        // Apply filter pipeline
        cv::Mat processedFrame;

        switch(activeFilter) {
            case FILTER_OPENCV_GREYSCALE:
                opencvGreyScale(frame, processedFrame);
                break;
            
            case FILTER_CUSTOM_GREYSCALE:
                customGreyScale(frame, processedFrame);
                break;
            
            case FILTER_SEPIA:
                sepia(frame, processedFrame);
                break;
            
            case FILTER_SOBEL_X:
                {
                    cv::Mat sobelResult;
                    sobelX3x3(frame, sobelResult);
                    cv::convertScaleAbs(sobelResult, processedFrame);
                }
                break;
            
            case FILTER_SOBEL_Y:
                {
                    cv::Mat sobelResult;
                    sobelY3x3(frame, sobelResult);
                    cv::convertScaleAbs(sobelResult, processedFrame);
                }
                break;
            
            case FILTER_GRADIENT_MAGNITUDE:
                {
                    cv::Mat sobelX, sobelY;
                    // Compute both Sobel X and Y
                    sobelX3x3(frame, sobelX);
                    sobelY3x3(frame, sobelY);
                    // Compute magnitude from both gradients
                    magnitude(sobelX, sobelY, processedFrame);
                }
                break;
            
            case FILTER_BLUR_QUANTIZE:
                blurQuantize(frame, processedFrame, 10);
                break;
            
            case FILTER_DEPTH_VIS_HOT:
                depthVisualizationHot(frame, processedFrame);
                break;
            
            case FILTER_DEPTH_VIS_COOL:
                depthVisualizationCool(frame, processedFrame);
                break;
            
            case FILTER_DEPTH_BLUR:
                depthBlur(frame, processedFrame);
                break;

            case FILTER_DEPTH_COLOR:
                depthColorGrade(frame, processedFrame);
                break;
            
            case FILTER_EMBOSS:
                emboss(frame, processedFrame);
                break;
            
            case FILTER_NEGATIVE:
                negative(frame, processedFrame);
                break;
            
            case FILTER_COLOR_ISOLATION_RED:
                isolateRed(frame, processedFrame);
                break;
            
            case FILTER_COLOR_ISOLATION_BLUE:
                isolateBlue(frame, processedFrame);
                break;
            
            case FILTER_COLOR_ISOLATION_GREEN:
                isolateGreen(frame, processedFrame);
                break;
            
            case FILTER_COLOR_ISOLATION_YELLOW:
                isolateYellow(frame, processedFrame);
                break;
            
            case FILTER_NONE:
            default:
                processedFrame = frame;
                break;
        }

        if (vignetteMode) {
            vignette(processedFrame, vignetteFrame, vignetteIntensity);
        } else {
            processedFrame.copyTo(vignetteFrame);
        }

        if (blurredMode) {
            blur5x5_2(vignetteFrame, blurredFrame);
        } else {
            vignetteFrame.copyTo(blurredFrame);
        }

        if (warmthIntensity > 0.0f) {
            warmth(blurredFrame, warmFrame, warmthIntensity);
        } else {
            blurredFrame.copyTo(warmFrame);
        }

        if (coolnessIntensity > 0.0f) {
            coolness(warmFrame, coolFrame, coolnessIntensity);
        } else {
            warmFrame.copyTo(coolFrame);
        }

        if (exposureValue != 1.0f) {
            exposure(coolFrame, exposedFrame, exposureValue);
        } else {
            coolFrame.copyTo(exposedFrame);
        }

        if (faceDetectMode) {
            // Convert to greyscale for face detection
            cv::Mat grey;
            cv::cvtColor(exposedFrame, grey, cv::COLOR_BGR2GRAY);
            
            // Detect faces
            std::vector<cv::Rect> faces;
            detectFaces(grey, faces);
            
            // Draw boxes on color image
            exposedFrame.copyTo(faceFrame);
            drawBoxes(faceFrame, faces);
        } else {
            exposedFrame.copyTo(faceFrame);
        }

        if (haloMode) {
            sparkleHalo(faceFrame, haloFrame);
        } else {
            faceFrame.copyTo(haloFrame);
        }

        if (mirrorMode) {
            mirror(haloFrame, displayFrame);
        } else {
            haloFrame.copyTo(displayFrame);
        }
        
        // Display the frame
        if (isRecording) {
            // Create copy with REC indicator for display only
            displayFrame.copyTo(displayWithIndicator);
            
            // Draw REC indicator (red circle + text)
            cv::circle(displayWithIndicator, cv::Point(30, 30), 12, cv::Scalar(0, 0, 255), -1);  // Red filled circle
            cv::circle(displayWithIndicator, cv::Point(30, 30), 14, cv::Scalar(255, 255, 255), 2);  // White outline
            cv::putText(displayWithIndicator, "REC", cv::Point(50, 38), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            
            cv::imshow(windowName, displayWithIndicator);
            
            // Write frame to video (without indicator)
            if (videoWriter.isOpened()) {
                videoWriter.write(displayFrame);
                recordedFrames++;
                
                // Print progress every 30 frames (~1 second)
                if (recordedFrames % 30 == 0) {
                    float seconds = recordedFrames / 30.0f;
                    printf("Recording: %.1f seconds (%d/%d frames)\n", seconds, recordedFrames, MAX_FRAMES);
                }
                
                // Auto-stop at max frames
                if (recordedFrames >= MAX_FRAMES) {
                    auto endTime = std::chrono::steady_clock::now();
                    float duration = std::chrono::duration<float>(endTime - recordingStartTime).count();
                    
                    videoWriter.release();
                    isRecording = false;
                    
                    printf("Recording stopped (max duration reached).\n");
                    printf("Video saved: %d frames (%.1f seconds)\n", recordedFrames, duration);
                    recordedFrames = 0;
                }
            }
        } else {
            // Normal display without indicator
            cv::imshow(windowName, displayFrame);
        }

        // See if there is a waiting keystroke
        key = cv::waitKey(30);
        
        // Handle keypresses
        switch(key) {
            case 'g':
                printf("Key Pressed: \"g\"\n");
                if (activeFilter == FILTER_OPENCV_GREYSCALE) {
                    activeFilter = FILTER_NONE;
                    printf("OpenCV greyscale mode: OFF\n");
                } else {
                    activeFilter = FILTER_OPENCV_GREYSCALE;
                    printf("OpenCV greyscale mode: ON\n");
                }
                break;
            

            case 'h':
                printf("Key Pressed: \"h\"\n");
                if (activeFilter == FILTER_CUSTOM_GREYSCALE) {
                    activeFilter = FILTER_NONE;
                    printf("Custom greyscale mode: OFF\n");
                } else {
                    activeFilter = FILTER_CUSTOM_GREYSCALE;
                    printf("Custom greyscale mode: ON\n");
                }
                break;
            

            case 'p':
                printf("Key Pressed: \"p\"\n");
                if (activeFilter == FILTER_SEPIA) {
                    activeFilter = FILTER_NONE;
                    printf("Sepia mode: OFF\n");
                } else {
                    activeFilter = FILTER_SEPIA;
                    printf("Sepia mode: ON\n");
                }
                break;
            
            
            case 'x':
                printf("Key Pressed: \"x\"\n");
                if (activeFilter == FILTER_SOBEL_X) {
                    activeFilter = FILTER_NONE;
                    printf("Sobel X mode: OFF\n");
                } else {
                    activeFilter = FILTER_SOBEL_X;
                    printf("Sobel X mode: ON\n");
                }
                break;
            
            
            case 'y':
                printf("Key Pressed: \"y\"\n");
                if (activeFilter == FILTER_SOBEL_Y) {
                    activeFilter = FILTER_NONE;
                    printf("Sobel Y mode: OFF\n");
                } else {
                    activeFilter = FILTER_SOBEL_Y;
                    printf("Sobel Y mode: ON\n");
                }
                break;


            case 'm':
                printf("Key Pressed: \"m\"\n");
                if (activeFilter == FILTER_GRADIENT_MAGNITUDE) {
                    activeFilter = FILTER_NONE;
                    printf("Gradient magnitude mode: OFF\n");
                } else {
                    activeFilter = FILTER_GRADIENT_MAGNITUDE;
                    printf("Gradient magnitude mode: ON\n");
                }
                break;
            
            
            case 'l':
                printf("Key Pressed: \"l\"\n");
                if (activeFilter == FILTER_BLUR_QUANTIZE) {
                    activeFilter = FILTER_NONE;
                    printf("Blur quantize mode: OFF\n");
                } else {
                    activeFilter = FILTER_BLUR_QUANTIZE;
                    printf("Blur quantize mode: ON (10 levels = 1000 colors)\n");
                }
                break;
            
            
            case 'n':
                printf("Key Pressed: \"n\"\n");
                if (activeFilter == FILTER_NEGATIVE) {
                    activeFilter = FILTER_NONE;
                    printf("Negative mode: OFF\n");
                } else {
                    activeFilter = FILTER_NEGATIVE;
                    printf("Negative mode: ON\n");
                }
                break;
            
            
            case 'z':
                printf("Key Pressed: \"z\"\n");
                if (activeFilter == FILTER_EMBOSS) {
                    activeFilter = FILTER_NONE;
                    printf("Emboss mode: OFF\n");
                } else {
                    activeFilter = FILTER_EMBOSS;
                    printf("Emboss mode: ON\n");
                }
                break;
            
            
            case 'd':
                printf("Key Pressed: \"d\"\n");
                // Cycle: None → Hot → Cool → Blur → Color → None
                if (activeFilter == FILTER_NONE || 
                    (activeFilter < FILTER_DEPTH_VIS_HOT || activeFilter > FILTER_DEPTH_COLOR)) {
                    // Switch to hot visualization
                    activeFilter = FILTER_DEPTH_VIS_HOT;
                    printf("Depth mode: Hot Visualization (INFERNO - red/orange)\n");
                } else if (activeFilter == FILTER_DEPTH_VIS_HOT) {
                    // Switch to cool visualization
                    activeFilter = FILTER_DEPTH_VIS_COOL;
                    printf("Depth mode: Cool Visualization (COOL - blue/cyan)\n");
                } else if (activeFilter == FILTER_DEPTH_VIS_COOL) {
                    // Switch to depth blur
                    activeFilter = FILTER_DEPTH_BLUR;
                    printf("Depth mode: Bokeh Effect (blur background)\n");
                } else if (activeFilter == FILTER_DEPTH_BLUR) {
                    // Switch to depth color grading
                    activeFilter = FILTER_DEPTH_COLOR;
                    printf("Depth mode: Color Grading (warm close, cool far)\n");
                } else {
                    // Turn off
                    activeFilter = FILTER_NONE;
                    printf("Depth mode: OFF\n");
                }
                break;
            
            
            case 'o':
                printf("Key Pressed: \"o\"\n");
                // Cycle: None → Red → Blue → Green → Yellow → None
                if (activeFilter == FILTER_NONE || 
                    (activeFilter < FILTER_COLOR_ISOLATION_RED || activeFilter > FILTER_COLOR_ISOLATION_YELLOW)) {
                    // Switch to red color mode
                    activeFilter = FILTER_COLOR_ISOLATION_RED;
                    printf("Color Mode: Red\n");
                } else if (activeFilter == FILTER_COLOR_ISOLATION_RED) {
                    // Switch to blue color mode
                    activeFilter = FILTER_COLOR_ISOLATION_BLUE;
                    printf("Color mode: Blue\n");
                } else if (activeFilter == FILTER_COLOR_ISOLATION_BLUE) {
                    // Switch to green color mode
                    activeFilter = FILTER_COLOR_ISOLATION_GREEN;
                    printf("Color mode: Green\n");
                } else if (activeFilter == FILTER_COLOR_ISOLATION_GREEN) {
                    // Switch to yellow color mode
                    activeFilter = FILTER_COLOR_ISOLATION_YELLOW;
                    printf("Color mode: Yellow\n");
                } else {
                    // Turn off
                    activeFilter = FILTER_NONE;
                    printf("Depth mode: OFF\n");
                }
                break;
            
            
            case 'v':
                printf("Key Pressed: \"v\"\n");
                vignetteMode = !vignetteMode;
                if (vignetteMode) {
                    printf("Vignette mode: ON (intensity: %.2f)\n", vignetteIntensity);
                } else {
                    printf("Vignette mode: OFF\n");
                }
                break;
            
            
            case '+':
            case '=':
                if (!vignetteMode) {
                    printf("Vignette mode is OFF - enable vignette mode to use this feature\n");
                    break;
                }
                vignetteIntensity += 0.1f;
                vignetteIntensity = vignetteIntensity > 2.0f ? 2.0f : vignetteIntensity;
                printf("Vignette intensity: %.2f\n", vignetteIntensity);
                if (vignetteIntensity == 2.0f) {
                    printf("Maximum vignette intensity\n");
                }
                break;
            

            case '-':
            case '_':
                if (!vignetteMode) {
                    printf("Vignette mode is OFF - enable vignette mode to use this feature\n");
                    break;
                }
                vignetteIntensity -= 0.1f;
                vignetteIntensity = vignetteIntensity < 0.0f ? 0.0f : vignetteIntensity;
                printf("Vignette intensity: %.2f\n", vignetteIntensity);
                if (vignetteIntensity <= 0.05f) {
                    printf("Minimum vignette intensity\n");
                }
                break;
            
            
            case 'b':
                printf("Key Pressed: \"b\"\n");
                blurredMode = !blurredMode;
                if (blurredMode) {
                    printf("Blur mode: ON\n");
                } else {
                    printf("Blur mode: OFF\n");
                }
                break;
            

            case 'w':
                printf("Key Pressed: \"w\"\n");
                warmthIntensity += 0.1f;
                warmthIntensity = warmthIntensity > 2.05f ? 0.0f : warmthIntensity;
                printf("Warmth intensity: %0.1f\n", warmthIntensity);
                break;
            
            
            case 'e':
                printf("Key Pressed: \"e\"\n");
                coolnessIntensity += 0.1f;
                coolnessIntensity = coolnessIntensity > 2.05f ? 0.0f : coolnessIntensity;
                printf("Coolness intensity: %0.1f\n", coolnessIntensity);
                break;
            
            
            case 'f':
                printf("Key Pressed: \"f\"\n");
                faceDetectMode = !faceDetectMode;
                if (faceDetectMode) {
                    printf("Face Detection mode: ON\n");
                } else {
                    printf("Face Detection mode: OFF\n");
                }
                break;
            
            
            case 't':
                printf("Key Pressed: \"t\"\n");
                haloMode = !haloMode;
                if (haloMode) {
                    printf("Halo mode: ON\n");
                } else {
                    printf("Halo mode: OFF\n");
                }
                break;
            
            
            case 'r':
                printf("Key Pressed: \"r\"\n");
                mirrorMode = !mirrorMode;
                if (mirrorMode) {
                    printf("Mirror mode: ON\n");
                } else {
                    printf("Mirror mode: OFF\n");
                }
                break;
            

            case '1':
                printf("Key Pressed: \"1\"\n");
                exposureValue += 0.1f;
                if (exposureValue > 1.5f) {
                    exposureValue = 1.5f;
                }
                printf("Exposure: %.2f\n", exposureValue);
                break;
            

            case '2':
                printf("Key Pressed: \"2\"\n");
                exposureValue -= 0.1f;
                if (exposureValue < 0.5f) {
                    exposureValue = 0.5f;
                }
                printf("Exposure: %.2f\n", exposureValue);
                break;
            
            
            case 'c':
                printf("Key Pressed: \"c\"\n");
                activeFilter = FILTER_NONE;
                blurredMode = false;
                vignetteMode = false;
                faceDetectMode = false;
                haloMode = false;
                vignetteIntensity = 0.5f;
                warmthIntensity = 0.0f;
                coolnessIntensity = 0.0f;
                exposureValue = 1.0f;
                printf("All filters cleared - back to normal color mode\n");
                break;
            
            
            case 'i':
                // Display image information
                printf("Key Pressed: \"i\"\n");
                printf("\n=========== Window Information ===========\n");
                printf("Resolution: %d x %d\n", refS.width, refS.height);
                printf("\n--- Active Filters ---\n");

                // Display primary filter
                printf("Primary Filter:  ");
                switch(activeFilter) {
                    case FILTER_OPENCV_GREYSCALE:
                        printf("OpenCV Greyscale\n");
                        break;
                    case FILTER_CUSTOM_GREYSCALE:
                        printf("Custom Greyscale\n");
                        break;
                    case FILTER_SEPIA:
                        printf("Sepia Tone\n");
                        break;
                    case FILTER_SOBEL_X:
                        printf("Sobel X\n");
                        break;
                    case FILTER_SOBEL_Y:
                        printf("Sobel Y\n");
                        break;
                    case FILTER_GRADIENT_MAGNITUDE:
                        printf("Gradient Magnitude\n");
                        break;
                    case FILTER_BLUR_QUANTIZE:
                        printf("Blur Quantize\n");
                        break;
                    case FILTER_DEPTH_VIS_HOT:
                        printf("Depth Visualization (Hot)\n");
                        break;
                    case FILTER_DEPTH_VIS_COOL:
                        printf("Depth Visualization (Cool)\n");
                        break;
                    case FILTER_DEPTH_BLUR:
                        printf("Depth Bokeh Effect\n");
                        break;
                    case FILTER_DEPTH_COLOR:
                        printf("Depth Color Grading\n");
                        break;
                    case FILTER_NEGATIVE:
                        printf("Negative\n");
                        break;
                    case FILTER_EMBOSS:
                        printf("Emboss\n");
                        break;
                    case FILTER_COLOR_ISOLATION_RED:
                        printf("Red Color Mode\n");
                        break;
                    case FILTER_COLOR_ISOLATION_BLUE:
                        printf("Blue Color Mode\n");
                        break;
                    case FILTER_COLOR_ISOLATION_GREEN:
                        printf("Green Color Mode\n");
                        break;
                    case FILTER_COLOR_ISOLATION_YELLOW:
                        printf("Yellow Color Mode\n");
                        break;
                    case FILTER_NONE:
                    default:
                        printf("None\n");
                        break;
                }

                // Display vignette status
                if (vignetteMode) {
                    printf("Vignette Intensity: %.2f)\n", vignetteIntensity);
                }

                // Display warmth status
                if (warmthIntensity > 0.0f) {
                    printf("Warmth Intensity: %.2f)\n", warmthIntensity);
                }

                // Display coolness status
                if (coolnessIntensity > 0.0f) {
                    printf("Coolness Intensity: %.2f)\n", coolnessIntensity);
                }

                // Display exposure info
                if (exposureValue != 1.0f) {
                    printf("Exposure: %.2f ", exposureValue);
                }

                // Display blurred status
                if (blurredMode) {
                    printf("Blurred Mode: ON\n");
                }
                
                //
                if (haloMode) {
                    printf("Halo Mode: ON\n");
                }

                // Display face detection status
                if (faceDetectMode) {
                    printf("Face Detection Mode: ON\n");
                }
                
                // Display mirror status
                if (mirrorMode) {
                    printf("Mirror Mode: ON\n");
                }
                
                printf("=========================================\n\n");

                break;
            
            
            case 's':
                // Save screenshot
                printf("Key Pressed: \"s\"\n");
                {
                    auto now = std::chrono::system_clock::now();
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

                    std::time_t t = std::chrono::system_clock::to_time_t(now);
                    std::tm tm_now;
                    localtime_s(&tm_now, &t);

                    char buffer[64];
                    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm_now);
                    std::string outputFilename = std::string("screenshot_") + buffer + "_" + std::to_string(ms.count()) + ".jpg";
                    if (cv::imwrite(outputFilename, displayFrame)) {
                        printf("Screenshot saved as: %s", outputFilename.c_str());
                        switch (activeFilter) {
                            case FILTER_OPENCV_GREYSCALE:
                                printf(" (OpenCV greyscale)");
                                break;

                            case FILTER_CUSTOM_GREYSCALE:
                                printf(" (Custom greyscale)");
                                break;
                            
                            case FILTER_SEPIA:
                                printf(" (Sepia tone)");
                                break;
                            
                            case FILTER_SOBEL_X:
                                printf(" (Sobel X)");
                                break;

                            case FILTER_SOBEL_Y:
                                printf(" (Sobel Y)");
                                break;
                            
                            case FILTER_GRADIENT_MAGNITUDE:
                                printf(" (Gradient Magnitude)");
                                break;
                            
                            case FILTER_BLUR_QUANTIZE:
                                printf(" (Blur Quantize)");
                                break;
                            
                            case FILTER_DEPTH_VIS_HOT:
                                printf(" (Depth Hot)");
                                break;
                            
                            case FILTER_DEPTH_VIS_COOL:
                                printf(" (Depth Cool)");
                                break;

                            case FILTER_DEPTH_BLUR:
                                printf(" (Depth Bokeh)");
                                break;

                            case FILTER_DEPTH_COLOR:
                                printf(" (Depth Color Grade)");
                                break;
                            
                            case FILTER_NEGATIVE:
                                printf(" (Negative)");
                                break;
                            
                            case FILTER_EMBOSS:
                                printf(" (Emboss)");
                                break;
                            
                            case FILTER_COLOR_ISOLATION_RED:
                                printf(" (Red Color Isolation)");
                                break;
                            
                            case FILTER_COLOR_ISOLATION_BLUE:
                                printf(" (Blue Color Isolation)");
                                break;
                            
                            case FILTER_COLOR_ISOLATION_GREEN:
                                printf(" (Green Color Isolation)");
                                break;
                            
                            case FILTER_COLOR_ISOLATION_YELLOW:
                                printf(" (Yellow Color Isolation)");
                                break;
                            
                            case FILTER_NONE:
                            default:
                                printf(" (Color)");
                                break;
                        }

                        if (vignetteMode) {
                            printf(" (Vignette: %.2f)", vignetteIntensity);
                        }

                        if (warmthIntensity > 0.0f) {
                            printf(" (Warmth: %.2f)", warmthIntensity);
                        }

                        if (coolnessIntensity > 0.0f) {
                            printf(" (Coolness: %.2f)", coolnessIntensity);
                        }

                        if (exposureValue != 1.0f) {
                            printf(" (Exposure: %.2f)", exposureValue);
                        }

                        if (blurredMode) {
                            printf(" (Blurred)");
                        }
                        
                        if (haloMode) {
                            printf(" (Halo)");
                        }

                        if (faceDetectMode) {
                            printf(" (Face Detection)");
                        }

                        if (mirrorMode) {
                            printf(" (Mirrored)");
                        }
                        printf("\n");
                    } else {
                        printf("Error: Failed to save screenshot\n");
                    }
                }
                break;
            
            
            case 'k':
                printf("Key Pressed: \"k\"\n");
                
                // Check if camera mode is being used
                if (!isCameraMode) {
                    printf("Video recording only available in video mode\n");
                    break;
                }
                
                if (!isRecording) {
                    // Start recording
                    // Show countdown
                    cv::Mat currentFrame = displayFrame;
                    if (!currentFrame.empty()) {
                        showCountdown(getProcessedFrame);
                    }
                    
                    // Generate filename
                    std::string filename = generateVideoFilename();
                    
                    // Initialize video writer
                    if (initializeVideoWriter(filename, cv::Size(refS.width, refS.height))) {
                        isRecording = true;
                        recordedFrames = 0;
                        recordingStartTime = std::chrono::steady_clock::now();
                        printf("Recording to: %s (max 30 seconds)\n", filename.c_str());
                    } else {
                        printf("ERROR: Failed to start recording\n");
                    }
                } else {
                    // Stop recording
                    float duration = recordedFrames / 30.0f;
                    
                    videoWriter.release();
                    isRecording = false;
                    
                    printf("Recording stopped.\n");
                    printf("Video saved: %d frames (%.1f seconds)\n", recordedFrames, duration);
                    recordedFrames = 0;
                }
                break;
            
            
            case '~':
            case '`':
                // Print the menu
                printf("Key Pressed: \"~\"\n");
                printf("Printing the menu\n\n");
                printMenu();
                break;
            

            case 'q':
                // Quit the application
                printf("Key Pressed: \"q\"\n");
                printf("Quitting application...\n");
                break;
            

            case -1:
                // No key pressed (timeout) - continue loop
                break;
            

            default:
                printf("Unknown command key pressed\n");
                break;
        }    
    }


    return 0;
}