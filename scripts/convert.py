#!/usr/bin/env python3
import os
import argparse
import sys
import requests
from urllib.parse import urlparse
from rknn.api import RKNN
import logging

# --- Configuration ---
INPUT_DIR = "/workspace/input_models"
OUTPUT_DIR = "/workspace/output_models"
DEFAULT_QUANT_DTYPE = "w16a16i_dfp"

# Default shapes (Width, Height) if --resolutions is not provided
# Extracted from the original DYNAMIC_INPUTS list
DEFAULT_SHAPES = [
    (1440, 404),
    (1440, 384),
    (1440, 320),
    (1536, 576),
    (1536, 512),
    (1536, 448),
]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Functions ---

def download_model(url: str, target_dir: str) -> str | None:
    """Downloads a model from a URL to the target directory."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_model.onnx" # Fallback filename

        target_path = os.path.join(target_dir, filename)
        os.makedirs(target_dir, exist_ok=True)

        logging.info(f"Downloading model from {url} to {target_path}...")
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info("Download complete.")
        return target_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download model from {url}: {e}")
        return None
    except OSError as e:
        logging.error(f"Failed to save model to {target_path}: {e}")
        return None

def parse_resolutions(res_string: str) -> list[tuple[int, int]] | None:
    """Parses a 'WxH,WxH,...' string into a list of (Width, Height) tuples."""
    shapes = []
    try:
        pairs = res_string.strip().split(',')
        for pair in pairs:
            if not pair: continue
            w_str, h_str = pair.strip().split('x')
            width = int(w_str)
            height = int(h_str)
            if width <= 0 or height <= 0:
                raise ValueError("Width and Height must be positive integers.")
            shapes.append((width, height))
        if not shapes:
            raise ValueError("No valid resolutions found in the string.")
        return shapes
    except ValueError as e:
        logging.error(f"Invalid format in resolutions string '{res_string}'. Use 'WxH,WxH,...'. Error: {e}")
        return None

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Convert ONNX model (local or URL) to multiple fixed-shape RKNN models."
    )
    p.add_argument(
        "--model_source",
        required=True,
        help="URL of the ONNX model or local filename (expected in input_models directory)."
    )
    p.add_argument(
        "--resolutions",
        help="Comma-separated list of target resolutions in WxH format (e.g., '1440x384,1536x512'). Defaults to predefined list if not specified."
    )
    p.add_argument(
        "--input_name",
        default="input",
        help="Name of the input node in the ONNX model (default: 'input'). Critical for some models."
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output from RKNN."
    )
    # Note: --output_dir is removed, using fixed OUTPUT_DIR
    # Note: quant_dtype is fixed for now, can be added as arg later
    return p.parse_args()

# --- Main Conversion Logic ---

def main():
    args = parse_args()
    onnx_model_path = None
    errors_occurred = False

    # 1. Determine and prepare ONNX model path
    source = args.model_source
    if source.startswith("http://") or source.startswith("https://"):
        onnx_model_path = download_model(source, INPUT_DIR)
        if not onnx_model_path:
            sys.exit(1) # Exit if download failed
    else:
        # Assume it's a local filename
        local_path = os.path.join(INPUT_DIR, source)
        if os.path.exists(local_path):
            onnx_model_path = local_path
            logging.info(f"Using local model: {onnx_model_path}")
        else:
            logging.error(f"Local model file not found: {local_path}")
            sys.exit(1)

    # 2. Determine target shapes
    target_shapes = []
    if args.resolutions:
        parsed_shapes = parse_resolutions(args.resolutions)
        if parsed_shapes:
            target_shapes = parsed_shapes
            logging.info(f"Using provided resolutions: {target_shapes}")
        else:
            sys.exit(1) # Exit if parsing failed
    else:
        target_shapes = DEFAULT_SHAPES
        logging.info(f"Using default resolutions: {target_shapes}")

    # 3. Get base model name for output files
    base_model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]

    # 4. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. Conversion loop
    quant_dtype = DEFAULT_QUANT_DTYPE # Fixed for now
    logging.info(f"Starting conversion for {len(target_shapes)} shapes...")

    for width, height in target_shapes:
        rknn = None # Ensure rknn is defined for finally block
        try:
            shape_str = f"{width}x{height}"
            logging.info(f"--- Converting shape: {shape_str} ---")

            # Create a new RKNN object for each shape (reliability over speed)
            rknn = RKNN(verbose=args.verbose)

            logging.info("[1/4] Configuring RKNN...")
            rknn.config(
                target_platform="rk3566",
                quantized_dtype=quant_dtype,
                optimization_level=2,
            )
            # Config doesn't return a useful value to check

            logging.info(f"[2/4] Loading ONNX model: {onnx_model_path}")
            ret = rknn.load_onnx(
                model=onnx_model_path,
                inputs=[args.input_name],
                input_size_list=[[1, 3, height, width]] # Note: H, W order
            )
            if ret != 0:
                raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
            logging.info("ONNX model loaded successfully.")

            logging.info("[3/4] Building RKNN model...")
            ret = rknn.build(do_quantization=False) # Keep False as discussed
            if ret != 0:
                raise RuntimeError(f"RKNN build failed with code {ret}")
            logging.info("RKNN model built successfully.")

            # Construct output path
            output_filename = f"{base_model_name}_{quant_dtype}_{shape_str}.rknn"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            logging.info(f"[4/4] Exporting RKNN model to: {output_path}")
            ret = rknn.export_rknn(output_path)
            if ret != 0:
                raise RuntimeError(f"RKNN export_rknn failed with code {ret}")
            logging.info(f"✅ Successfully exported RKNN model for shape {shape_str}!")

        except Exception as e:
            logging.error(f"❌ FAILED to convert shape {shape_str}: {e}")
            errors_occurred = True
            # Continue to the next shape

        finally:
            if rknn:
                rknn.release()
                logging.debug(f"RKNN object released for shape {shape_str}.")

    # 6. Final status
    logging.info("--- Conversion process finished ---")
    if errors_occurred:
        logging.warning("Some shapes failed to convert. Check logs above.")
        sys.exit(1)
    else:
        logging.info("All shapes converted successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
