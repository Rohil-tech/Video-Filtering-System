# CS 5330 - PR-CV - Assignment 1
Rohil Kulshreshtha
January 24, 2026

## Project Overview
Real-time video and image filtering application with 27+ interactive effects.

## File Structure
- `src/` - Source files (filter.cpp, filterApp.cpp, vidDisplay.cpp, imgDisplay.cpp, timeBlur.cpp, faceDetect.cpp)
- `include/` - Header files (filter.h, filterApp.h, faceDetect.h, DA2Network.hpp)
- `bin/` - Executables and model files (haarcascade, model_fp16.onnx, etc.)
- `data/` - Test images (For testing image mode)

## Dependencies
- OpenCV 4.12.0
- ONNX Runtime 1.17.1
- CMake 3.15+
- C++17 compiler

## Build Instructions
```bash
cmake -B build -S .
cmake --build build
```

## Running (From Project Root Directory)
```bash
# Video mode (camera)
./bin/vidDisplay.exe

# Image mode
./bin/imgDisplay.exe data/image.jpg

# Performance testing
./bin/timeBlur.exe data/image.jpg
```

## Key Controls
### Primary Filters
- 'g' - OpenCV Greyscale
- 'h' - Custom Greyscale
- 'p' - Sepia Tone
- 'x' - Sobel X
- 'y' - Sobel Y
- 'm' - Gradient Magnitude
- 'l' - Blur Quantize
- 'n' - Negative
- 'z' - Emboss
- 'd' - Cycle Depth Modes
- 'o' - Cycle Color Isolation Modes

### Secondary Effects
- 'v' - Vignette (+/- to adjust intensity) (0.0-2.0)
- 'b' - Gaussian Blur - 5x5
- 'w' - Warmth (cycle 0.0-2.0)
- 'e' - Coolness (cycle 0.0-2.0)
- 'f' - Face Detection
- 't' - Sparkle Halo (above detected faces)
- 'r' - Mirror/Flip
- '1' - Increase Exposure/Brightness
- '2' - Decrease Exposure/Brightness

### Actions
- 'c' - Clear All Filters
- 'i' - Display Info About Active Filters and Effects
- 's' - Save screenshot
- 'k' - Record Video (Live Mode Only)
- '~' - Check the Menu for Command Options
- 'q' - Quit

## Video Demonstrations
Recording with Filters Demo:
https://northeastern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f1d7799f-e3cf-453e-bb0a-b3de001418f1

## Known Issues
- First time initialization of depth estimation may take a few secondss