/**
 * Rohil Kulshreshtha
 * January 18, 2026
 * CS 5330 - PR-CV - Assignment 1
 * 
 * Implementation of image filter functions.
*/

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "faceDetect.h"
#include "filter.h"
#include "DA2Network.hpp"

static DA2Network* g_da2_network = nullptr;

/**
 * Color Isolation: Keeps pixels of a target color, converts rest to greyscale.
 * 
 * Method: Detects pixels within a certain hue range and keeps them in color.
 * Everything else becomes greyscale. Creates a dramatic selective color effect.
 */
int colorIsolation(cv::Mat &src, cv::Mat &dst, int targetHue, int hueRange) {
    // Convert to HSV for easier color selection
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    // Create greyscale version
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    // Back to 3-channel
    cv::cvtColor(grey, grey, cv::COLOR_GRAY2BGR);
    
    // Start with greyscale
    grey.copyTo(dst);
    
    // Copy color pixels that match target hue
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *hsvPtr = hsv.ptr<cv::Vec3b>(i);
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            unsigned char h = hsvPtr[j][0];
            unsigned char s = hsvPtr[j][1];
            
            // Hue (0-179 in OpenCV)
            // Saturation (0-255)
            // Check if pixel is within target hue range and has enough saturation
            int hueDiff = abs(h - targetHue);
            // Handle hue wrapping (red is at 0 and 180)
            if (hueDiff > 90) {
                hueDiff = 180 - hueDiff;
            }
            
            // Keep color if within hue range and saturated enough
            // s > 30 filters out greyish colors
            if (hueDiff <= hueRange && s > 30) {
                // Keep original color
                dstPtr[j] = srcPtr[j];
            }
            // Otherwise it stays greyscale (already copied)
        }
    }
    
    return 0;
}


/**
 * Convenience function for common color selection for color isolation
 * Red hue ~0, range ±15
 */
int isolateRed(cv::Mat &src, cv::Mat &dst) {
    return colorIsolation(src, dst, 0, 15);
}

/**
 * Convenience function for common color selection for color isolation
 * Blue hue ~110, range ±15
 */
int isolateBlue(cv::Mat &src, cv::Mat &dst) {
    return colorIsolation(src, dst, 110, 15);
}

/**
 * Convenience function for common color selection for color isolation
 * Green hue ~60, range ±15
 */
int isolateGreen(cv::Mat &src, cv::Mat &dst) {
    return colorIsolation(src, dst, 60, 15);
}

/**
 * Convenience function for common color selection for color isolation
 * // Yellow hue ~30, range ±15
 */
int isolateYellow(cv::Mat &src, cv::Mat &dst) {
    return colorIsolation(src, dst, 30, 15);
}


/**
 * Adds a sparkly halo above detected faces.
 * Uses face detection to locate head, then adds sparkles in an arc above.
 * Optional: Uses depth to make sparkles appear in front/behind head.
 */
int sparkleHalo(cv::Mat &src, cv::Mat &dst) {
    src.copyTo(dst);
    
    // Detect faces
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    detectFaces(grey, faces);
    
    // For each detected face, add sparkles above
    for(const auto &face : faces) {
        // Calculate halo position (above head)
        int centerX = face.x + face.width / 2;
        int haloY = face.y - face.height / 4;
        int haloRadius = face.width / 2;
        
        // Add sparkles in an arc above the head
        // Arc from 30° to 150°
        for(int angle = 30; angle <= 150; angle += 15) {
            float rad = angle * 3.14159f / 180.0f;
            int sparkleX = centerX + (int)(haloRadius * cos(rad));
            int sparkleY = haloY - (int)(haloRadius * 0.5f * sin(rad));
            
            // Check bounds
            if (sparkleX >= 3 && sparkleX < src.cols - 3 && 
                sparkleY >= 3 && sparkleY < src.rows - 3) {
                
                // Draw a sparkle (small bright cross/star)
                // Yellow sparkle
                cv::Scalar sparkleColor(50, 200, 255);
                
                // Center point (bright)
                dst.at<cv::Vec3b>(sparkleY, sparkleX) = cv::Vec3b(sparkleColor[0], sparkleColor[1], sparkleColor[2]);
                
                // Four-point star
                dst.at<cv::Vec3b>(sparkleY-1, sparkleX) = cv::Vec3b(sparkleColor[0], sparkleColor[1], sparkleColor[2]);
                dst.at<cv::Vec3b>(sparkleY+1, sparkleX) = cv::Vec3b(sparkleColor[0], sparkleColor[1], sparkleColor[2]);
                dst.at<cv::Vec3b>(sparkleY, sparkleX-1) = cv::Vec3b(sparkleColor[0], sparkleColor[1], sparkleColor[2]);
                dst.at<cv::Vec3b>(sparkleY, sparkleX+1) = cv::Vec3b(sparkleColor[0], sparkleColor[1], sparkleColor[2]);
                
                // Diagonal points (softer)
                cv::Scalar dimColor(25, 100, 128);
                dst.at<cv::Vec3b>(sparkleY-1, sparkleX-1) = cv::Vec3b(dimColor[0], dimColor[1], dimColor[2]);
                dst.at<cv::Vec3b>(sparkleY-1, sparkleX+1) = cv::Vec3b(dimColor[0], dimColor[1], dimColor[2]);
                dst.at<cv::Vec3b>(sparkleY+1, sparkleX-1) = cv::Vec3b(dimColor[0], dimColor[1], dimColor[2]);
                dst.at<cv::Vec3b>(sparkleY+1, sparkleX+1) = cv::Vec3b(dimColor[0], dimColor[1], dimColor[2]);
            }
        }
    }
    
    return 0;
}


/**
 * Applies an emboss effect to the image.
 * Uses Sobel X and Sobel Y signed outputs and computes a dot product with
 * direction vector (0.7071, 0.7071) to create a 3D relief appearance.
 */
int emboss(cv::Mat &src, cv::Mat &dst) {
    // Allocate destination image
    dst.create(src.size(), src.type());

    // Compute Sobel X and Y
    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    // Valid convolution
    for(int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s *sxPtr = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syPtr = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);

        for(int j = 1; j < src.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {
                // Dot product with direction vector (0.7071, 0.7071)
                float value = 0.7071f * sxPtr[j][c] + 0.7071f * syPtr[j][c];
                // Offset to mid-gray
                value += 128.0f;

                // Clip to 0-255
                value = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);

                dstPtr[j][c] = (unsigned char)value;
            }
        }
    }

    return 0;
}


/**
 * Inverts colors of the input image (negative effect).
 * Each channel is replaced with 255 - original value.
 */
int negative(cv::Mat &src, cv::Mat &dst) {
    // Allocate destination image
    dst.create(src.size(), src.type());

    // Process each row
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        // Process each pixel
        for(int j = 0; j < src.cols; j++) {
            dstPtr[j][0] = 255 - srcPtr[j][0];
            dstPtr[j][1] = 255 - srcPtr[j][1];
            dstPtr[j][2] = 255 - srcPtr[j][2];
        }
    }

    return 0;
}


/**
 * Applies depth-based blur (bokeh effect).
 * Uses Depth Anything V2 network to estimate depth.
 */
int depthBlur(cv::Mat &src, cv::Mat &dst) {
    if (g_da2_network == nullptr) {
        printf("Loading Depth Anything V2 model...\n");
        g_da2_network = new DA2Network("bin/model_fp16.onnx");
        printf("Model loaded!\n");
    }
    
    float scale_factor = 384.0f / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0f ? 1.0f : scale_factor;
    
    // Set network input with original frame
    g_da2_network->set_input(src, scale_factor);
    
    // Get depth map at original size
    cv::Mat depthMap;
    g_da2_network->run_network(depthMap, src.size());
    
    // Create heavily blurred version
    cv::Mat blurred;
    // blur5x5_2(src, blurred);
    // More blurring than the one created as part of the assignment
    cv::GaussianBlur(src, blurred, cv::Size(21, 21), 5.0);
    
    // Allocate destination
    dst.create(src.size(), src.type());
    
    // Focus threshold: objects with depth > 180 stay sharp
    int focusThreshold = 180;
    
    // Blend original and blurred based on depth
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *blurPtr = blurred.ptr<cv::Vec3b>(i);
        unsigned char *depthPtr = depthMap.ptr<unsigned char>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            unsigned char d = depthPtr[j];
            
            // Calculate blend factor based on depth
            float blendFactor;
            if (d > focusThreshold) {
                blendFactor = 1.0f;  // Keep sharp
            } else {
                blendFactor = (float)d / focusThreshold;  // Gradual blur
            }
            
            // Blend between original and blurred
            for(int c = 0; c < 3; c++) {
                float sharp = srcPtr[j][c];
                float blur = blurPtr[j][c];
                dstPtr[j][c] = (unsigned char)(sharp * blendFactor + blur * (1.0f - blendFactor));
            }
        }
    }
    
    return 0;
}


/**
 * Applies depth-based color grading.
 * Uses Depth Anything V2 network to estimate depth.
 */
int depthColorGrade(cv::Mat &src, cv::Mat &dst) {
    // Get depth map
    if (g_da2_network == nullptr) {
        printf("Loading Depth Anything V2 model...\n");
        g_da2_network = new DA2Network("bin/model_fp16.onnx");
        printf("Model loaded!\n");
    }
    
    float scale_factor = 384.0f / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0f ? 1.0f : scale_factor;
    
    g_da2_network->set_input(src, scale_factor);
    
    cv::Mat depthMap;
    g_da2_network->run_network(depthMap, src.size());
    
    // Allocate destination
    dst.create(src.size(), src.type());
    
    // Apply color grading based on depth
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        unsigned char *depthPtr = depthMap.ptr<unsigned char>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            unsigned char d = depthPtr[j];
            float b = srcPtr[j][0];
            float g = srcPtr[j][1];
            float r = srcPtr[j][2];
            
            float newBlue, newGreen, newRed;
            
            if (d > 170) {
                // Close: Warm tones
                float warmth = (d - 170) / 85.0f;
                newBlue  = b * (1.0f - warmth * 0.3f);
                newGreen = g * (1.0f + warmth * 0.1f);
                newRed   = r * (1.0f + warmth * 0.4f);
            } else if (d < 85) {
                // Far: Cool tones
                float coolness = (85 - d) / 85.0f;
                newBlue  = b * (1.0f + coolness * 0.4f);
                newGreen = g * (1.0f + coolness * 0.1f);
                newRed   = r * (1.0f - coolness * 0.3f);
            } else {
                // Mid: Natural
                newBlue = b;
                newGreen = g;
                newRed = r;
            }
            
            // Clip to 0-255
            newBlue  = newBlue < 0 ? 0 : (newBlue > 255 ? 255 : newBlue);
            newGreen = newGreen < 0 ? 0 : (newGreen > 255 ? 255 : newGreen);
            newRed   = newRed < 0 ? 0 : (newRed > 255 ? 255 : newRed);
            
            dstPtr[j][0] = (unsigned char)newBlue;
            dstPtr[j][1] = (unsigned char)newGreen;
            dstPtr[j][2] = (unsigned char)newRed;
        }
    }
    
    return 0;
}


/**
 * Applies depth estimation and creates a colored depth map visualization (Red-Heatmap-INFERNO).
 * Uses Depth Anything V2 network to estimate depth.
 */
int depthVisualizationHot(cv::Mat &src, cv::Mat &dst) {
    // Initialize network on first use
    if (g_da2_network == nullptr) {
        printf("Loading Depth Anything V2 model (this may take a moment)...\n");
        g_da2_network = new DA2Network("bin/model_fp16.onnx");
        printf("Model loaded successfully!\n");
    }
    
    // Scale factor to keep image reasonable size for network
    // Smaller = faster, but less accurate
    float scale_factor = 384.0f / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0f ? 1.0f : scale_factor;
    
    // Set network input
    g_da2_network->set_input(src, scale_factor);
    
    // Run network to get depth map
    cv::Mat depth;
    g_da2_network->run_network(depth, src.size());
    
    // Apply colormap for visualization (INFERNO: blue=far, red=close)
    cv::applyColorMap(depth, dst, cv::COLORMAP_INFERNO);
    
    return 0;
}


/**
 * Applies depth estimation and creates a colored depth map visualization (Blue-Coolmap-OCEAN).
 * Uses Depth Anything V2 network to estimate depth.
 */
int depthVisualizationCool(cv::Mat &src, cv::Mat &dst) {
    // Initialize network on first use
    if (g_da2_network == nullptr) {
        printf("Loading Depth Anything V2 model (this may take a moment)...\n");
        g_da2_network = new DA2Network("bin/model_fp16.onnx");
        printf("Model loaded successfully!\n");
    }
    
    // Scale factor to keep image reasonable size for network
    // Smaller = faster, but less accurate
    float scale_factor = 384.0f / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0f ? 1.0f : scale_factor;
    
    // Set network input
    g_da2_network->set_input(src, scale_factor);
    
    // Run network to get depth map
    cv::Mat depth;
    g_da2_network->run_network(depth, src.size());
    
    // Apply colormap for visualization (INFERNO: cyan=far, purple=close)
    cv::applyColorMap(depth, dst, cv::COLORMAP_COOL);
    
    return 0;
}


/**
 * Increases or descreases exposure on the image.
 */
int exposure(cv::Mat &src, cv::Mat &dst, float exposure) {
  // Allocate destination image
  dst.create(src.size(), src.type());

  for(int i=0;i<src.rows;i++) {
    cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    for(int j=0;j<src.cols;j++) {
      float b = ptr[j][0];
      float g = ptr[j][1];
      float r = ptr[j][2];

      // Multiply by exposure
      b *= exposure;
      g *= exposure;
      r *= exposure;

      // Clipping to 255
      b = b > 255 ? 255.0 : b;
      g = g > 255 ? 255.0 : g;
      r = r > 255 ? 255.0 : r;

      dptr[j][0] = (unsigned char)b;
      dptr[j][1] = (unsigned char)g;
      dptr[j][2] = (unsigned char)r;
    }
  }

  return(0);
}


/**
 * Applies a warmth filter by enhancing red/orange tones.
 */
int warmth(cv::Mat &src, cv::Mat &dst, float intensity) {
    // Allocate destination image
    dst.create(src.size(), src.type());
    
    // Process each pixel
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            // Get BGR values
            unsigned char b = srcPtr[j][0];
            unsigned char g = srcPtr[j][1];
            unsigned char r = srcPtr[j][2];
            
            // Apply warmth transformation by boosting red, slightly boosting green and reducing blue
            float newBlue  = b - (b * 0.2f * intensity);         // Reduce blue
            float newGreen = g + ((255 - g) * 0.1f * intensity); // Slight boost green
            float newRed   = r + ((255 - r) * 0.3f * intensity); // Strong boost red
            
            // Clip to 255
            newBlue  = newBlue < 0 ? 0 : (newBlue > 255 ? 255 : newBlue);
            newGreen = newGreen > 255.0 ? 255.0 : newGreen;
            newRed   = newRed > 255.0 ? 255.0 : newRed;
            
            // Write to destination
            dstPtr[j][0] = (unsigned char)newBlue;
            dstPtr[j][1] = (unsigned char)newGreen;
            dstPtr[j][2] = (unsigned char)newRed;
        }
    }

    return 0;
}


/**
 * Applies a coolness filter by enhancing blue/cyan tones.
 */
int coolness(cv::Mat &src, cv::Mat &dst, float intensity) {
    // Allocate destination image
    dst.create(src.size(), src.type());
    
    // Process each pixel
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            // Get BGR values
            unsigned char b = srcPtr[j][0];
            unsigned char g = srcPtr[j][1];
            unsigned char r = srcPtr[j][2];
            
            // Apply warmth transformation by boosting blue, slightly boosting green and reducing red
            float newBlue   = b + ((255 - b) * 0.3f * intensity); // Strong boost blue
            float newGreen = g + ((255 - g) * 0.1f * intensity);  // Slight boost green
            float newRed  = r - (r * 0.2f * intensity);           // Reduce red
            
            // Clip to 255
            newBlue   = newBlue > 255.0 ? 255.0 : newBlue;
            newGreen = newGreen > 255.0 ? 255.0 : newGreen;
            newRed  = newRed < 0 ? 0 : (newRed > 255 ? 255 : newRed);
            
            // Write to destination
            dstPtr[j][0] = (unsigned char)newBlue;
            dstPtr[j][1] = (unsigned char)newGreen;
            dstPtr[j][2] = (unsigned char)newRed;
        }
    }

    return 0;
}


/*
  Applies blur followed by color quantization.
  
  Blur the image using 5x5 Gaussian blur and then quantize each color channel to specified levels
  
  Quantization process:
  - Bucket size b = 255 / levels
  - For each pixel value x:
    - xt = x / b (integer division - which bucket)
    - xf = xt * b (bucket representative value)
  
  Example with levels=10:
  - b = 255 / 10 = 25
  - Input: 37  → xt = 37/25 = 1 → xf = 1*25 = 25
  - Input: 128 → xt = 128/25 = 5 → xf = 5*25 = 125
  
  Result: Creates a posterized/cartoon effect with levels^3 total colors.
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    cv::Mat blurred;
    
    // Blur the image using existing blur function
    blur5x5_2(src, blurred);
    
    // Quantize the blurred image
    // Allocate destination
    dst.create(blurred.size(), blurred.type());
    
    // Calculate bucket size
    int bucketSize = 255 / levels;
    
    // Process each row
    for(int i = 0; i < blurred.rows; i++) {
        cv::Vec3b *srcPtr = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        // Process each pixel
        for(int j = 0; j < blurred.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // Get original value
                unsigned char x = srcPtr[j][c];
                
                // Determine which bucket, then get bucket value
                int xt = x / bucketSize;            // Which bucket (integer division)
                unsigned char xf = xt * bucketSize; // Bucket representative value
                
                // Store quantized value
                dstPtr[j][c] = xf;
            }
        }
    }
    
    return 0;
}


/**
 * Computes gradient magnitude from Sobel X and Y results using Euclidean distance: magnitude = sqrt(sx * sx + sy * sy)
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Allocate destination as 8-bit unsigned, 3-channel
    dst.create(sx.size(), CV_8UC3);

    //Process each row
    for(int i = 0; i < sx.rows; i++) {
        cv::Vec3s *sxPtr = sx.ptr<cv::Vec3s>(i);  // 16-bit signed
        cv::Vec3s *syPtr = sy.ptr<cv::Vec3s>(i);  // 16-bit signed
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);  // 8-bit unsigned
        
        // Process each pixel
        for(int j = 0; j < sx.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // Get gradient values from Sobel X and Y
                short gx = sxPtr[j][c];
                short gy = syPtr[j][c];
                
                // Compute magnitude using Euclidean distance
                float mag = sqrt((float)((gx * gx) + (gy * gy)));
                
                // Clip max value to 255
                mag = mag > 255.0f ? 255.0f : mag;
                
                // Store as unsigned char
                dstPtr[j][c] = (unsigned char)mag;
            }
        }
    }

    return 0;
}


/**
 * Applies a 3x3 Sobel X filter (horizontal gradient detection).
 * Detects vertical edges (positive right direction).
 * Sobel X kernel: [-1  0  1]
 *                 [-2  0  2]
 *                 [-1  0  1]
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    // Allocate destination as 16-bit signed, 3-channel
    dst.create(src.size(), CV_16SC3);

    // Apply filter to inner pixels (skip 1-pixel border)
    for(int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b *srcPtrUp = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *srcPtrMid = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *srcPtrDn = src.ptr<cv::Vec3b>(i + 1);
        // Initialize dst pointer as 16-bit signed
        cv::Vec3s *dstPtr = dst.ptr<cv::Vec3s>(i);
        
        for(int j = 1; j < src.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {
                // Apply Sobel X kernel
                int value = srcPtrUp[j-1][c] * (-1) + srcPtrUp[j+1][c] * 1
                    + srcPtrMid[j-1][c] * (-2) + srcPtrMid[j+1][c] * 2
                    + srcPtrDn[j-1][c] * (-1) + srcPtrDn[j+1][c] * 1;
                
                // Store as signed short
                dstPtr[j][c] = (short)value;
            }
        }
    }

    return 0;
}


/**
 * Applies a 3x3 Sobel Y filter (vertical gradient detection).
 * Detects horizontal edges (positive up direction).
 * Sobel Y kernel: [ 1  2  1]
 *                 [ 0  0  0]
 *                 [-1 -2 -1]
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    // Allocate destination as 16-bit signed, 3-channel
    dst.create(src.size(), CV_16SC3);
    
    // Apply filter to inner pixels (skip 1-pixel border)
    for(int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b *srcPtrUp = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *srcPtrDn = src.ptr<cv::Vec3b>(i + 1);
        // Initialize dst pointer as 16-bit signed
        cv::Vec3s *dstPtr = dst.ptr<cv::Vec3s>(i);
        
        for(int j = 1; j < src.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {
                // Apply Sobel Y kernel
                int value = srcPtrUp[j-1][c] * 1 + srcPtrUp[j][c] * 2 + srcPtrUp[j+1][c] * 1
                    + srcPtrDn[j-1][c] * (-1) + srcPtrDn[j][c] * (-2) + srcPtrDn[j+1][c] * (-1);
                
                // Store as signed short
                dstPtr[j][c] = (short)value;
            }
        }
    }

    return 0;
}


/**
 * Implement a 5x5 Guassian filter as
 * 
 *   1  2   4  2  1 
 *   2  4   8  4  2
 *   4  8  16  8  4
 *   2  4   8  4  2
 *   1  2   4  2  1
 * 
 * This version uses the at<> method of cv::Mat
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // Copy the src to the dst to allocate the dst
    src.copyTo(dst);

    // Define the 5x5 Gaussian kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    // Valid convolution
    // Apply filter to inner pixels (skip 2-pixel border)
    for(int i=2;i<src.rows-2;i++) {
        for(int j=2;j<src.cols-2;j++) {
            // Process each color channel
            for(int c=0;c<3;c++) {
                int value = 0;

                // Apply 5x5 kernel
                for(int ki = 0; ki < 5; ki++) {
                    for(int kj = 0; kj < 5; kj++) {
                        value += src.at<cv::Vec3b>(i - 2 + ki, j - 2 + kj)[c] * kernel[ki][kj];
                    }
                }
                
                // Normalize by kernel sum (100)
                dst.at<cv::Vec3b>(i, j)[c] = value / 100;
            }
        }
    }

    return 0;
}


/**
 * Implement a 5x5 Guassian filter as separable filters
 * 
 *                      1
 *                      2
 *   1 2 4 2 1    *     4
 *                      2
 *                      1
 * 
 * This version uses the ptr<> method of cv::Mat
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    cv::Mat tmp;

    // Copy the src to a intermediate result
    src.copyTo(tmp);

    // Loop that convolves with 1x5 filter [1 2 4 2 1]
    // Apply filter to inner pixels (skip 2-pixel border)
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tptr = tmp.ptr<cv::Vec3b>(i);
        
        for(int j = 2; j < src.cols - 2; j++) {
            for(int c = 0; c < 3; c++) {
                // Apply horizontal kernel: [1 2 4 2 1]
                int value = sptr[j-2][c] * 1 + sptr[j-1][c] * 2 + sptr[j][c] * 4 + sptr[j+1][c] * 2 + sptr[j+2][c] * 1;
                
                // Normalize by horizontal kernel sum (10)
                tptr[j][c] = value / 10;
            }
        }
    }

    // Allocate the dst and copy the intermediate result
    tmp.copyTo( dst );

    // Loop that convolves with 1x5 filter [1 2 4 2 1]^T
    // Apply filter to inner pixels (skip 2-pixel border)
    for(int i = 2; i < src.rows - 2; i++) {
        // Get pointers for the 5 rows needed for the kernel
        cv::Vec3b *tptr_up2 = tmp.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *tptr_up1 = tmp.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *tptr_mid = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3b *tptr_dn1 = tmp.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *tptr_dn2 = tmp.ptr<cv::Vec3b>(i+2);
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
        
        for(int j = 0; j < src.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // Apply vertical kernel: [1 2 4 2 1]^T
                int value = tptr_up2[j][c] * 1 + tptr_up1[j][c] * 2 + tptr_mid[j][c] * 4 + tptr_dn1[j][c] * 2 + tptr_dn2[j][c] * 1;
                
                // Normalize by vertical kernel sum (10)
                dptr[j][c] = value / 10;
            }
        }
    }

    return 0;
}


/**
 * Applies a vignette effect (darkening towards the edges) to the video feed in any filter.
 */
int vignette(cv::Mat &src, cv::Mat &dst, float intensity) {
    // Allocate destination image
    dst.create(src.size(), src.type());
    
    // Calculate center of image
    float centerX = src.cols / 2.0f;
    float centerY = src.rows / 2.0f;

    // Calculate maximum distance from center to corner
    float maxDist = sqrt(centerX * centerX + centerY * centerY);

    // Process each row
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        // Process each pixel
        for(int j = 0; j < src.cols; j++) {
            // Calculate distance from current pixel to center
            float dx = j - centerX;
            float dy = i - centerY;
            float distance = sqrt(dx * dx + dy * dy);
            
            // Normalize distance to [0, 1] range
            float normalizedDist = distance / maxDist;
            
            // Calculate vignette factor
            // factor = 1.0 at center, decreases towards edges based on intensity
            float factor = 1.0f - (intensity * normalizedDist * normalizedDist);
            
            // Clipping factor to prevent negative values
            factor = factor < 0.0f ? 0.0f : factor;
            
            // Apply vignette to each channel
            for(int c = 0; c < 3; c++) {
                float value = srcPtr[j][c] * factor;
                dstPtr[j][c] = (unsigned char)value;
            }
        }
    }

    return 0;
}

/**
 * Applies a sepia tone filter to the video feed to create an antique photograph effect.
 */
int sepia(cv::Mat &src, cv::Mat &dst) {
    // Allocate destination image with same size and type as source
    dst.create(src.size(), src.type());

    // Process each row
    for(int i = 0; i < src.rows; i++) {
        // Get pointers to current row in source and destination
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        // Process each pixel
        for(int j = 0; j < src.cols; j++) {
            // Get the original BGR values
            unsigned char b = srcPtr[j][0];
            unsigned char g = srcPtr[j][1];
            unsigned char r = srcPtr[j][2];
            
            // Calculate new values using transformation matrix
            float newBlue  = 0.272f * r + 0.534f * g + 0.131f * b;
            float newGreen = 0.349f * r + 0.686f * g + 0.168f * b;
            float newRed   = 0.393f * r + 0.769f * g + 0.189f * b;
            
            // Clip the values to 255 to prevent overflow
            newBlue = newBlue > 255 ? 255.0 : newBlue;
            newGreen = newGreen > 255 ? 255.0 : newGreen;
            newRed = newRed > 255 ? 255.0 : newRed;
            
            // Write to destination (BGR order)
            dstPtr[j][0] = (unsigned char)newBlue;
            dstPtr[j][1] = (unsigned char)newGreen;
            dstPtr[j][2] = (unsigned char)newRed;
        }
    }

    return 0;
}


/**
 * Applies standard custom greyscale conversion to the video feed.
 */
int customGreyScale(cv::Mat &src, cv::Mat &dst) {
    // Allocate destination image with same size and type as source
    dst.create(src.size(), src.type());
    
    // Process each row
    for(int i = 0; i < src.rows; i++) {
        // Get pointers to current row in source and destination
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        // Process each column pixel
        for(int j = 0; j < src.cols; j++) {
            // Get BGR values
            unsigned char b = srcPtr[j][0];
            unsigned char g = srcPtr[j][1];
            unsigned char r = srcPtr[j][2];
            
            // Custom greyscale: use minimum of R, G, B channels
            unsigned char grey = std::min({b, g, r});
            
            // Set all three channels to the same grey value
            dstPtr[j][0] = grey;
            dstPtr[j][1] = grey;
            dstPtr[j][2] = grey;
        }
    }
    
    return 0;
}


/**
 * Applies standard OpenCV greyscale conversion to the video feed.
 */
int opencvGreyScale(cv::Mat &src, cv::Mat &dst) {
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, dst, cv::COLOR_GRAY2BGR);

    return 0;
}


/**
 * Mirrors/flips the video feed along the vertical axis (left becomes right and right becomes left).
 */
int mirror(cv::Mat &src, cv::Mat &dst) {
    src.copyTo(dst);
    
    // Process each row
    for(int i = 0; i < dst.rows; i++) {
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        
        // Process first half of columns (swap with second half)
        for(int j = 0; j < dst.cols / 2; j++) {
            int mirrorCol = dst.cols - 1 - j;
            
            // Swap pixels
            cv::Vec3b temp = dstPtr[j];
            dstPtr[j] = dstPtr[mirrorCol];
            dstPtr[mirrorCol] = temp;
        }
    }
    
    return 0;
}