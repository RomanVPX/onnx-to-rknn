# ONNX to RKNN Converter

Automated converter for ONNX models (particularly ESRGAN) to RKNN format for Rockchip NPU (rk3566).

## Overview

This project addresses several challenges:

1. **RKNN SDK Complexity** - Uses Docker with pinned dependency versions for reliable conversion
2. **Dynamic Inputs Bug** - Works around RKNN SDK's `dynamic_inputs` issue by generating fixed-resolution models
3. **Manual Process** - Automates the creation of multiple RKNN models from a single ONNX file

## Usage

### GitHub Actions (Recommended)

1. Go to Actions → "ONNX to RKNN Conversion"
2. Click "Run workflow"
3. Fill in the parameters:
   - ONNX model URL
   - Target resolutions (comma-separated, e.g., `1440x384,1536x512`)
   - Input tensor name (optional, default: `input`)
4. The converted models will be available:
   - In the workflow run artifacts
   - In an automatically created GitHub Release
   - Via permanent download URLs

### Local Docker

```bash
docker run --rm -v $(pwd)/input_models:/workspace/input_models \
                -v $(pwd)/output_models:/workspace/output_models \
                ghcr.io/OWNER/REPO/rknn-converter:latest \
                --model_source https://example.com/model.onnx \
                --resolutions 1440x384,1536x512 \
                --input_name input
```

#### Parameters

- `--model_source`: URL or local path (in ./input_models) to ONNX model
- `--resolutions`: Comma-separated list of resolutions (WxH,WxH,...)
- `--input_name`: Input tensor name (default: input)
- `-v`: Enable verbose output

### Testing with docker-compose

1. Place ONNX model in ./input_models
2. Configure parameters in docker-compose.yml
3. Run:
```bash
docker-compose up
```

## Project Structure

```
.
├── .github/
│   └── workflows/          # GitHub Actions workflows
│       └── convert.yml     # Main conversion workflow
├── Dockerfile             # Image with RKNN SDK and deps
├── docker-compose.yml     # Quick start config
├── scripts/
│   └── convert.py        # Core conversion script
├── input_models/         # Input ONNX models
└── output_models/        # Converted RKNN models
```

## Technical Details

### Core Features
- Automated conversion via GitHub Actions
- Docker image in GitHub Container Registry
- Permanent model URLs via GitHub Releases
- Registry-based cache for faster builds

### Model Details
- **Input:** ONNX format (tested with ESRGAN)
- **Output:** RKNN format for rk3566
- **Quantization:** w16a16i_dfp without re-quantization
- **Output Names:** `model_w16a16i_dfp_WIDTHxHEIGHT.rknn`

### GitHub Integration
- **Container Registry:** `ghcr.io/OWNER/REPO/rknn-converter:tag`
- **Release URLs:** `github.com/OWNER/REPO/releases/download/TAG/model_WxH.rknn`
- **Permissions:** Requires write access to packages and contents

## Known Issues

- RKNN SDK has a critical bug with `dynamic_inputs` that causes segfaults with ESRGAN models. This is worked around by generating separate models for each resolution.
- GitHub Container Registry requires lowercase repository names
- Model conversion can be resource-intensive (CPU/RAM)

## Example Usage

Convert Real-ESRGAN x2 model:
```bash
# Via GitHub Actions:
URL: https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.onnx
Resolutions: 1440x320,1440x384
Input name: input

# Via Docker:
docker run --rm -v ./input_models:/workspace/input_models \
              -v ./output_models:/workspace/output_models \
              ghcr.io/OWNER/REPO/rknn-converter:latest \
              --model_source https://huggingface.co/.../RealESRGAN_x2plus.onnx \
              --resolutions 1440x320,1440x384
