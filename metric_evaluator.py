import argparse
import os
import numpy as np
import OpenEXR
import Imath
import math
import scipy as spy
from PIL import Image
from scipy.stats import entropy
import tifffile
import tqdm
from PIL import Image
import io


# global-local metric that contributes to the final metric basing on how "eroded" the terrain is
# TODO: normalize the metric by computing upper-bound for the given dimensionality
def evaluate_erosion_score(heightmap: np.ndarray):
    height, width = heightmap.shape

    n = 0
    mean = 0.0
    m2 = 0.0

    for y in range(height):
        for x in range(width):
            center = heightmap[y, x]

            max_delta = 0.0

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx = x + dx
                    ny = y + dy

                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue

                    neighbor = heightmap[ny, nx]
                    delta = abs(neighbor - center)

                    if delta > max_delta:
                        max_delta = delta
            n += 1
            d = max_delta - mean
            mean += d / n
            d2 = max_delta - mean
            m2 += d * d2

    variance = (m2 / n) if n > 1 else 0.0
    std_dev = math.sqrt(variance)

    erosion_score = std_dev / mean if mean != 0 else 0.0

    return erosion_score


# TODO: normalize the metric
# TODO: think how the magnitude can be included
def evaluate_gradient_score(heightmap: np.ndarray, subdomainSize: int):
    gy, gx = np.gradient(heightmap)
    height, width = heightmap.shape

    average_gradients = []
    spans = []

    for y in tqdm.tqdm(range(int(height / subdomainSize))):
        for x in range(int(width / subdomainSize)):

            start_x = x * subdomainSize
            start_y = y * subdomainSize

            end_x = start_x + subdomainSize
            end_y = start_y + subdomainSize

            local_gradients = np.array(
                [
                    gx[start_y:end_y, start_x:end_x],
                    gy[start_y:end_y, start_x:end_x],
                ]
            )

            norm = np.linalg.norm(local_gradients, axis=0) + 1e-12
            normalized_gradients = local_gradients / norm

            g = normalized_gradients.reshape(2, -1).T
            dot_matrix = g @ g.T
            min_dot = np.clip(np.min(dot_matrix), -1.0, 1.0)
            max_dot_angle = math.acos(min_dot)

            spans.append(max_dot_angle)

            avg_gradient = [
                np.mean(local_gradients[0]),
                np.mean(local_gradients[1]),
            ]
            avg_gradient /= np.linalg.norm(avg_gradient) + 1e-12
            average_gradients.append(avg_gradient)

    average_gradients = np.array(average_gradients)

    try:
        g = average_gradients.reshape(2, -1).T
        dot_matrix = g @ g.T
        min_dot = np.clip(np.min(dot_matrix), -1.0, 1.0)
        max_global_angle = math.acos(min_dot)
    except Exception as e:
        print(f"Exception occurred during gradient score evaluation: {e}")
        print("Doing things in plain-old style.")
        max_global_angle = 0
        for i in tqdm.tqdm(range(len(average_gradients))):
            grad_1 = (average_gradients[i, 0], average_gradients[i, 1])
            for j in range(i, len(average_gradients)):
                grad_2 = (average_gradients[j, 0], average_gradients[j, 1])
                dot = max(
                    -1.0,
                    min(1.0, (grad_1[0] * grad_2[0] + grad_1[1] * grad_2[1])),
                )
                angle = math.acos(dot)
                if angle > max_global_angle:
                    max_global_angle = angle

    average_angle = np.mean(spans)
    gradient_score = max_global_angle / max(1e-12, average_angle)

    return gradient_score


def evaluate_fractal_score(heightmap: np.ndarray):
    pass


def compressed_size_png(arr):
    img = Image.fromarray(arr)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return len(buffer.getvalue())


def compressed_size_jpg(arr, quality=95):
    img = Image.fromarray(arr)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)

    return len(buffer.getvalue())


def shannon_entropy(arr):
    values, counts = np.unique(arr, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)


def evaluate_global_aesthetic_measure(heightmap: np.ndarray):
    height, width = heightmap.shape

    copy_heightmap = heightmap.copy()
    copy_heightmap = np.round(255.0 * (copy_heightmap / np.max(copy_heightmap))).astype(
        np.uint8
    )

    texel_entropy = shannon_entropy(copy_heightmap)
    initial_information_content = (height * width) * texel_entropy

    heightmap_png_size = compressed_size_png(copy_heightmap)
    heightmap_jpg_size = compressed_size_jpg(copy_heightmap)

    zurek_png = (
        initial_information_content - heightmap_png_size
    ) / initial_information_content
    zurek_jpg = (
        initial_information_content - heightmap_jpg_size
    ) / initial_information_content

    return (zurek_png, zurek_jpg)


def read_exr_grayscale(path: str) -> np.ndarray:
    exr = OpenEXR.InputFile(path)

    header = exr.header()
    data_window = header["dataWindow"]

    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel = exr.channel("Y", pt)

    heightmap = np.frombuffer(channel, dtype=np.float32)
    heightmap = heightmap.reshape((height, width))

    return heightmap


def read_jpg_grayscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def read_tiff_grayscale(path: str, normalize: bool = False) -> np.ndarray:
    arr = tifffile.imread(path)

    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    tiff_as_np = arr.astype(np.float32)
    if normalize:
        max_value = np.max(tiff_as_np)
        tiff_as_np /= max_value
    return tiff_as_np


def process_heightmap(heightmap: np.ndarray, filename: str, chunk_size: int):
    print("-" * 32)
    print(f"Processing heightmap {filename}")
    print("Shape:", heightmap.shape)
    print("Min:", heightmap.min(), "Max:", heightmap.max())

    erosion_score = evaluate_erosion_score(heightmap)
    print(f"Erosion score: {erosion_score}")

    gradient_score = evaluate_gradient_score(heightmap, chunk_size)
    print(f"Gradient score: {gradient_score}")

    zurek_png, zurek_jpg = evaluate_global_aesthetic_measure(heightmap)
    print(f"GAM: png : ({zurek_png}) jpg : ({zurek_jpg})")


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
            heightmap = read_exr_grayscale(path)
        elif fmt in ("jpg", "jpeg"):
            heightmap = read_jpg_grayscale(path)
        elif fmt in ("tif", "tiff"):
            heightmap = read_tiff_grayscale(path)
        else:
            raise RuntimeError("Unexpected format")

        process_heightmap(heightmap, filename, args.chunk_size)

        del heightmap


if __name__ == "__main__":
    main()
