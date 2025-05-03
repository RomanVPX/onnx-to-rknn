#!/usr/bin/env python3
import os
import argparse
import sys
import requests
from urllib.parse import urlparse
from rknn.api import RKNN
import logging
import onnx

# --- Configuration ---
INPUT_DIR = "/workspace/input_models"
OUTPUT_DIR = "/workspace/output_models"

# "w8a8","w4a16", "w8a16", "w4a8", "w16a16i" and "w16a16i_dfp" — Rockchip documentation
# w16a16i_dfp(*), w16a16i(*), w8a8 — works with rk3566
# (*) — not supported by rk3566 according to Rockchip documentation, but works in practice
# w4a16, w8a16 — not supported by rk3566, like really not supported
# w4a8 — exists only in documentation, not in rknn api
# w8a16 is forced when `quantize_weight` (which is "about to be deprecated") is set to True
DEFAULT_QUANT_DTYPE = "w8a8"

TARGET_PLATFORMS = {
    "rv1103", "rv1103b", "rv1106", "rv1106b", "rv1126b",
    "rk2118", "rk3562", "rk3566", "rk3568", "rk3576", "rk3588"
}
DEFAULT_TARGET_PLATFORM = "rk3566"

# Default shapes (Width, Height) if --resolutions is not provided
DEFAULT_SHAPES = [
    (1440, 320), (1440, 384), (1440, 404),
    (1536, 448), (1536, 512), (1536, 576)
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
        "--target_platform",
        default=DEFAULT_TARGET_PLATFORM,
        choices=TARGET_PLATFORMS,
        help=f"Target platform for RKNN model. Default: {DEFAULT_TARGET_PLATFORM}."
    )
    p.add_argument(
        "--resolutions",
        help="Comma-separated list of target resolutions in WxH format (e.g., '1440x384,1536x512'). Defaults to predefined list if not specified."
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output from RKNN."
    )
    return p.parse_args()


def get_onnx_input_name(model_path: str) -> str | None:
    """Loads ONNX model and returns the name of the first input tensor."""
    try:
        logging.info(f"Loading ONNX model {model_path} to determine input name...")
        onnx_model = onnx.load(model_path)
        if not onnx_model.graph.input:
            logging.error("ONNX model graph has no inputs!")
            return None
        # For ESRGAN, there is usually one input. We take the name of the first one.
        input_name = onnx_model.graph.input[0].name
        input_shape = [d.dim_param if d.dim_param else d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        logging.info(f"Detected input name: '{input_name}' with shape hint: {input_shape}")

        if len(onnx_model.graph.input) > 1:
            logging.warning(f"Model has multiple inputs ({len(onnx_model.graph.input)}). Using the first one: '{input_name}'. Ensure this is correct.")

        return input_name

    except Exception as e:
        logging.error(f"Failed to load or parse ONNX model {model_path}: {e}")
        return None


# --- Main Conversion Logic ---
def main():
    args = parse_args()
    onnx_model_path = None
    errors_occurred = False

    # 0. Validate target platform
    logging.info(f"Target platform: {args.target_platform}")
    if args.target_platform.lower() not in TARGET_PLATFORMS:
        logging.error(f"Invalid target platform '{args.target_platform}'. Supported platforms: {TARGET_PLATFORMS}.")
        sys.exit(1)

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

    # 2. Determine ONNX input name
    onnx_input_name = get_onnx_input_name(onnx_model_path)
    if not onnx_input_name:
        logging.error("Could not determine ONNX input name. Aborting.")
        sys.exit(1)

    # 3. Determine target shapes
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

    # 4. Get base model name for output files
    base_model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]

    # 5. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 6. Conversion loop
    quant_dtype = DEFAULT_QUANT_DTYPE # Doeesn't affect when do_quantization=False
    logging.info(f"Starting conversion for {len(target_shapes)} shapes...")

    for width, height in target_shapes:
        rknn = None
        try:
            shape_str = f"{width}x{height}"
            logging.info(f"--- Converting shape: {shape_str} ---")
            rknn = RKNN(verbose=args.verbose)

            logging.info("[1/4] Configuring RKNN...")
            rknn.config(
                target_platform=args.target_platform,
                quantized_dtype=quant_dtype,
                optimization_level=2
            )
            # Config doesn't return a useful value to check

            logging.info(f"[2/4] Loading ONNX model: {onnx_model_path}, detected input name: '{onnx_input_name}'...")
            ret = rknn.load_onnx(
                model=onnx_model_path,
                inputs=[onnx_input_name],
                input_size_list=[[1, 3, height, width]] # Note: H, W order
            )
            if ret != 0: raise RuntimeError(f"RKNN load_onnx failed with code {ret}")
            logging.info("ONNX model loaded successfully.")

            logging.info("[3/4] Building RKNN model...")
            ret = rknn.build(do_quantization=False)
            if ret != 0: raise RuntimeError(f"RKNN build failed with code {ret}")
            logging.info("RKNN model built successfully.")

            output_filename = f"{base_model_name}_{args.target_platform}_{shape_str}.rknn"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            logging.info(f"[4/4] Exporting RKNN model to: {output_path}")
            ret = rknn.export_rknn(output_path)
            if ret != 0: raise RuntimeError(f"RKNN export_rknn failed with code {ret}")
            logging.info(f"✅ Successfully exported RKNN model for shape {shape_str}!")

        except Exception as e:
            logging.error(f"❌ FAILED to convert shape {shape_str}: {e}")
            errors_occurred = True
        finally:
            if rknn:
                rknn.release()
                logging.debug(f"RKNN object released for shape {shape_str}.")

    # 7. Final status
    logging.info("--- Conversion process finished ---")
    if errors_occurred:
        logging.warning("Some shapes failed to convert. Check logs above.")
        sys.exit(1)
    else:
        logging.info("All shapes converted successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
