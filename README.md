# L2S

This repository provides the web client for the Learn2See project, built with Vite.

## Run Locally

1. Ensure you have the latest version of npm installed.
2. `npm install`
3. `npm run dev` to start the development server.
4. `npm run build` to generate production assets.

Pre-trained ONNX models should be placed in `public/models` so they are available at `/models/` at runtime.
Legacy Python utilities are preserved under `python/` for reference but are not served by the application.

## Code Overview

New to the project? Start with [doc/code-overview.md](doc/code-overview.md) for a tour of the TypeScript modules and how they fit together.
