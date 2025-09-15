# TypeScript Code Overview

This document summarizes the main modules in the Learn2See web client to help new contributors navigate the codebase.

## Entry Point

- **src/main.ts** – Bootstraps the application, initializes the ONNX runtime, wires up the UI, and starts monitoring backend availability.

## Core Components

- **src/GazeDetector.ts** – Coordinates webcam capture, landmark detection, gaze prediction, and target presentation. Emits events with prediction results and training feedback.
- **src/GazeElement.ts** – Custom HTML element whose position represents either the predicted gaze point or a training target.
- **src/UI.ts** – Holds references to DOM elements and reusable UI helpers.

## Controllers

- **src/controllers/GazeSession.ts** – Manages the lifecycle of gaze detection and optional model training sessions.
- **src/controllers/TrainingPage.ts** – UI controller for the training view.

## Services

- **src/services/DataAcquisitionService.ts** – Drives the moving target used to collect labelled gaze samples.
- **src/services/InputHandler.ts** – Handles keyboard shortcuts for saving models, toggling data acquisition, and starting calibration.
- **src/services/NotificationService.ts** – Lightweight notification helper.

## Runtime and Training Utilities

- **src/runtime/WebOnnxAdapter.ts** – Wraps `onnxruntime-web` and exposes an async API for running PCA and MLP models in the browser.
- **src/runtime/TrainableOnnx.ts** – Adds save/export support to ONNX sessions during training.
- **src/training/Trainer.ts** – Orchestrates incremental training of the gaze model in the browser.

## Helpers

- **src/util/** – Miscellaneous utility functions such as coordinate conversions, buffer helpers, and navigation helpers.

This overview is intentionally high level. Refer to source files for detailed implementation notes.
