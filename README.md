# ONNX to RKNN Converter (for Rockchip NPU)

[![Build and Convert](https://github.com/RomanVPX/onnx-to-rknn/actions/workflows/convert.yml/badge.svg)](https://github.com/RomanVPX/onnx-to-rknn/actions/workflows/convert.yml)

A Dockerized tool to convert ONNX models (specifically targeting image upscalers like ESRGAN) to the RKNN format for Rockchip NPUs (tested on RK3566).

## Why Does This Exist? (The Problem)

Converting models for Rockchip NPUs using their official `rknn-toolkit2` is... an experience. Its documentation is a masterpiece of obfuscation, and key features like `dynamic_input` (essential for models that should handle variable input sizes, like image upscalers) are **fundamentally broken** for many (if not all) architectures, leading to crashes (`Segmentation fault`) or garbage output during inference.

After much suffering, apparently the only reliable method found was to generate **separate RKNN models for each specific, fixed input resolution** needed, bypassing the buggy dynamic shape handling. Doing this manually for multiple resolutions is tedious and error-prone.

## What It Does (The Solution)

This tool takes a *single* ONNX model (which should ideally support dynamic input dimensions, although the tool can sometimes force a fixed size even on static ONNX files) and generates *multiple* RKNN models, each compiled for a specific, fixed input resolution (e.g., 1280x256, 1920x1080).

It uses the `rknn-toolkit2` library inside a Docker container with pinned dependencies to ensure a consistent environment, leveraging the (working) `input_size_list` parameter during `rknn.load_onnx()` to force the desired input shape for each conversion.

## Features

*   **Dockerized:** Packages everything into a neat container with pinned dependencies (`rknn-toolkit2==2.3.0`, specific `onnx`, `numpy`, etc.) for reproducibility.
*   **Handles URLs and Local Files:** Provide an HTTP(S) URL or a local filename for the source ONNX model.
*   **Multiple Resolutions:** Generates **multiple** RKNN models for a list of specified **fixed** input resolutions (e.g., `1440x384`, `1536x512`).
*   **Automatic Input Name Detection:** Uses the `onnx` library to find the input tensor name automatically.
*   **Fixed Shape Output:** Generates reliable RKNN models by targeting *fixed* input shapes, avoiding the buggy `dynamic_input` feature of the SDK.
*   **GitHub Actions Workflow:** Includes a workflow (Check the `.github/workflows` directory) to automatically:
    *   Build the converter Docker image (caching layers in GHCR).
    *   Download an ONNX model from a URL.
    *   Convert the model for specified resolutions *in parallel*.
    *   Create a GitHub Release containing the generated `.rknn` files.

## How to Use

### 1. GitHub Actions Workflow

This repository includes a reusable GitHub Actions workflow (`.github/workflows/convert.yml`) to automate the process entirely within GitHub.

1.  **Fork** this repository.
2.  **Go to the "Actions" tab** of your repository fork.
3.  **Select** the "ONNX to RKNN Conversion" workflow.
4.  **Click** "Run workflow".
5.  **Fill in the inputs:**
      *   `URL of ONNX model to convert`: URL to the source ONNX model.
      *   `Comma-separated list of target resolutions`: Just like it says
      *   `Target platform for conversion`: Target Rockchip platform (e.g., `RK3566`, `RK3588`). Default: `RK3566`.
      *   `Custom release tag`: _(Optional)_ A custom tag for your release. If left empty, the release tag will be `models-<run_id>`
      *   `Custom release name`: _(Optional)_ A custom name for your release. If left empty, the release name will look like `<source_model_filename> for <target_platform>`
6.  **Click** "Run workflow".
7.  **Wait:** The workflow will:
      *   Build and push the converter Docker image to your repository's GHCR (it will be used as a cache in the next runs).
      *   Download the ONNX model.
      *   Run the conversion for each specified resolution in parallel jobs.
      *   Collect all generated `.rknn` files.
      *   Create a new GitHub Release tagged `models-<run_id>` containing the `.rknn` files as assets and release notes.
8.  **Download:** Go to the Releases page of your repository and download the `.rknn` files attached to the newly created release.

### 2. Local Conversion (via Docker Compose)

This is useful for testing or converting models locally.

**Prerequisites:**
*   Docker and Docker Compose installed.

**Setup:**
1.  Clone this repository.
2.  (Optional) Place your local ONNX model file inside the `./input_models/` directory.
3.  Edit the `command:` section in `docker-compose.yml` to specify your conversion parameters:
     *   `--model_source`: Change this to your ONNX model's URL or the **filename** of a model placed in `./input_models/`.
     *   `--resolutions` (optional): Provide a comma-separated list of `WidthxHeight` resolutions you need (e.g., `"1440x320,1536x576"`). If omitted, it uses a default list defined in the script.
     *   `--target_platform` (optional): Specify the target chip (e.g., `RK3588`). Defaults to `RK3566`.
     *   `-v` or `--verbose` (optional): Add this for more detailed logs from `rknn-toolkit2`.

4.  **Run the conversion:**
    ```bash
    docker-compose up --build
    ```
    *   `--build` is only needed the first time or if you change `Dockerfile` or `convert.py`.
    *   Watch the logs. The script will download the model (if needed) and convert it for each specified resolution.
5.  **Find your models:** The generated `.rknn` files will appear in the `./output_models/` directory on your host machine. The filenames will include the target platform and resolution (e.g., `YourModel_rk3566_1440x320.rknn`).
6.  **Stop the container:**
       ```bash
       docker-compose down
       ```

**Example `command` in `docker-compose.yml`:**
   _Using a URL and specific resolutions to run on the RK388 platform:_
```yaml
command: >
   --model_source https://huggingface.co/some_user/some_model/resolve/main/model.onnx
   --resolutions 1280x256,1024x512
   --target_platform RK3588
   --verbose
```
_Using a local file and default resolutions:_
```yaml
command: >
   --model_source my_local_esrgan.onnx
```

## Troubleshooting / Notes

*   **Segmentation Faults during Inference:** If the generated `.rknn` model *still* crashes the NPU during inference (especially with large resolutions), it might be hitting hardware memory limits or bugs in the specific model/operator conversion for that size. Try converting for slightly smaller resolutions.
*   **Model Size:** Don't be surprised if the "fixed-shape" `.rknn` files generated by this method are larger than expected. The conversion process isn't always perfectly optimized.

## Known Rockchip SDK Quirks Encountered

*   **`dynamic_input` is broken/misleading:** Don't use it in `rknn.config`. It doesn't provide true dynamic shapes and often leads to errors or crashes.
*   **`input_size_list` is key:** Specifying a fixed shape via `input_size_list` in `rknn.load_onnx()` is the reliable way to generate a model for a *specific* target resolution, even from an ONNX model that *claims* to have dynamic axes.
*   **`inputs` parameter is critical:** You *must* provide the correct input tensor name(s) via the `inputs` parameter in `rknn.load_onnx()`, otherwise `input_size_list` might be ignored, defaulting to 128x128 or 64x64 input. (This script detects the name automatically).
*   **Weird parameters:** Some `rknn.config` options like `compress_weight` might increase model size, and `quantize_weight` is deprecated in a way that makes it unusable on some platforms. Stick to basics unless you enjoy pain.
*   **`librknnrt.so` location:** The Python toolkit expects the native runtime library (`librknnrt.so`) in weird hardcoded paths like `/usr/lib64`. The NPU service (`upscaler.py` in the other project) uses a workaround to copy/link this library at container start. The *converter* doesn't need this library as it only uses the toolkit's Python parts.

## Contributing

Feel free to open issues or PRs if you find bugs or ways to improve this clusterfuck.
