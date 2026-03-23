import evaluationlib as elib
import numpy as np
from PIL import Image
import argparse
import os

def process_heightmap(
    heightmap: np.ndarray, filename: str, chunk_size: int, division_depth: int
):
    quantized_heightmap = elib.quantize_heightmap(heightmap, 255.0)
    normalized_heightmap = elib.normalize_heightmap(heightmap)

    print("-" * 32)
    print(f"Processing heightmap {filename}")
    print("Shape:", heightmap.shape)
    print("Min:", heightmap.min(), "Max:", heightmap.max())

    erosion_score = elib.evaluate_erosion_score(normalized_heightmap)
    print(f"Erosion score: {erosion_score}")

    gradient_score = elib.evaluate_gradient_score(normalized_heightmap, chunk_size)
    print(f"Gradient score: {gradient_score}")

    fractal_dimension, beta, mse = elib.evaluate_fractal_score(heightmap)
    print(f"Fractal dimenison: {fractal_dimension}")
    print(f"Noise beta exponent: {beta}. Fit MSE: {mse}")

    zurek_png, zurek_lzma, zurek_zlib = elib.evaluate_global_aesthetic_measure(
        quantized_heightmap
    )
    print(
        f"GAM:\n  png : ({zurek_png}) \n  zlib : ({zurek_zlib}) \n  lzma : ({zurek_lzma})"
    )

    (order_png, order_lzma, order_zlib), split_visualization = (
        elib.evaluate_composite_aesthetics_measure(
            heightmap,
            division_depth,
            [elib.compressed_size_png, elib.compressed_size_lzma, elib.compressed_size_zlib],
        )
    )
    Image.fromarray(split_visualization).save(
        f"./{'.'.join(filename.split('.')[:-1])}_split.png", format="PNG"
    )

    print(
        f"CAM:\n  png : ({order_png}) \n  zlib : ({order_zlib}) \n  lzma : ({order_lzma})"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps.")

    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing heightmaps",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="exr",
        help="Heightmap format (exr, jpg, jpeg, tif, tiff)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Should such a need arise, the heightmap will be split into chunks of size NxN",
    )

    parser.add_argument(
        "--division-depth",
        type=int,
        required=True,
        help="The depth of the order-analyzing tree",
    )

    args = parser.parse_args()

    directory = args.directory
    fmt = args.format.lower()

    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")

    extensions = {
        "exr": [".exr"],
        "jpg": [".jpg", ".jpeg"],
        "jpeg": [".jpg", ".jpeg"],
        "tif": [".tif", ".tiff"],
        "tiff": [".tif", ".tiff"],
    }

    if fmt not in extensions:
        raise ValueError("Unsupported format")

    files = sorted(
        f
        for f in os.listdir(directory)
        if any(f.lower().endswith(ext) for ext in extensions[fmt])
    )

    for filename in files:
        path = os.path.join(directory, filename)

        if fmt == "exr":
            heightmap = elib.read_exr_grayscale(path)
        elif fmt in ("jpg", "jpeg"):
            heightmap = elib.read_jpg_grayscale(path)
        elif fmt in ("tif", "tiff"):
            heightmap = elib.read_tiff_grayscale(path)
        else:
            raise RuntimeError("Unexpected format")

        process_heightmap(heightmap, filename, args.chunk_size, args.division_depth)

        del heightmap


if __name__ == "__main__":
    main()
