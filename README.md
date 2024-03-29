# Custom Mediapipe Calculators
Custom calculators for Mediapipe framework.

## Archived (No longer to be continued)
Please check [mp_proctor](https://github.com/sawthiha/mp_proctor.git) as it organizes all the calculators, graphs and the demo application.

## Features
- Utility
    - Landmark Standardization Calculator
- Face orientation
    - orientation Detector
    - orientation-to-RenderData
- Eye Blink
    - Blink Detector
    - Blink-to-RenderData

## Requirement
- Mediapipe v0.8.10.2 (Simply checkout on [this commit](https://github.com/google/mediapipe/commit/63e679d9))

## Installation
You simply put this source directory as submodule into `mediapipe/calculators/custom` or anywhere else inside `mediapipe` where you can reference from bazel. Or, simply run the following commands from the `mediapipe` root directory.

```bash
git submodule add https://github.com/sawthiha/mediapipe_calculators.git mediapipe/calculators/custom
```