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
import zlib
import tqdm
import io

JPEG_MAX_DIM = 65500

# TODO: check if I need to discard compression headers from the compressed heightmaps


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

    for y in range(int(height / subdomainSize)):
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


def compressed_size_zlib(arr):
    return len(zlib.compress(arr.tobytes()))


# assumes the values are strictly positive
def quantize_heightmap(heightmap: np.ndarray):
    copy_heightmap = heightmap.copy()
    copy_heightmap = np.round(255.0 * (copy_heightmap / np.max(copy_heightmap))).astype(
        np.uint8
    )
    return copy_heightmap


def concatenate_domains_rect(sub1: np.ndarray, sub2: np.ndarray):
    h1, w1 = sub1.shape
    h2, w2 = sub2.shape

    if h1 == h2 and w1 + w2 < JPEG_MAX_DIM:
        return np.concatenate([sub1, sub2], axis=1)

    if w1 == w2 and h1 + h2 < JPEG_MAX_DIM:
        return np.concatenate([sub1, sub2], axis=0)

    flat = np.concatenate([sub1.ravel(), sub2.ravel()])

    N = flat.size

    if N >= JPEG_MAX_DIM**JPEG_MAX_DIM:
        raise ValueError("The heightmap pieces to stitch are way too huge")

    width = int(np.sqrt(N))

    while width > 1 and N % width != 0:
        width -= 1

    height = N // width

    if height >= JPEG_MAX_DIM:
        width = JPEG_MAX_DIM

        while width > 1 and N % width != 0:
            width -= 1

        height = N // width

    if height >= JPEG_MAX_DIM or width >= JPEG_MAX_DIM:
        return None

    return flat.reshape(height, width)


def shannon_entropy(arr):
    values, counts = np.unique(arr, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)


def evaluate_global_aesthetic_measure(heightmap: np.ndarray):
    height, width = heightmap.shape

    copy_heightmap = quantize_heightmap(heightmap)

    texel_entropy = shannon_entropy(copy_heightmap)
    initial_information_content = (height * width) * texel_entropy

    heightmap_png_size = compressed_size_png(copy_heightmap)
    heightmap_jpg_size = compressed_size_jpg(copy_heightmap)
    heightmap_zlib_size = compressed_size_zlib(copy_heightmap)

    zurek_png = (
        initial_information_content - heightmap_png_size
    ) / initial_information_content
    zurek_jpg = (
        initial_information_content - heightmap_jpg_size
    ) / initial_information_content
    zurek_zlib = (
        initial_information_content - heightmap_zlib_size
    ) / initial_information_content

    return (zurek_png, zurek_jpg, zurek_zlib)


def mutual_information_from_histograms(h1, h2):
    """
    Mutual information between two discrete distributions defined by histograms.
    """
    n1 = h1.sum()
    n2 = h2.sum()

    if n1 == 0 or n2 == 0:
        return 0.0

    p1 = h1 / n1
    p2 = h2 / n2

    p = (h1 + h2) / (n1 + n2)

    mask1 = p1 > 0
    mask2 = p2 > 0

    h_left = -np.sum(p1[mask1] * np.log2(p1[mask1]))
    h_right = -np.sum(p2[mask2] * np.log2(p2[mask2]))

    mask = p > 0
    h_total = -np.sum(p[mask] * np.log2(p[mask]))

    return h_left + h_right - h_total


def best_split(heightmap, y0, y1, x0, x1, min_rel_size):
    """
    Find best vertical or horizontal split maximizing MI.
    """

    sub = heightmap[y0:y1, x0:x1]
    h, w = sub.shape

    total_hist = np.bincount(sub.ravel(), minlength=256)

    best_mi = -1
    best_split = None

    min_h = int(h * min_rel_size)
    min_w = int(w * min_rel_size)

    # vertical splits
    if w >= 2 * min_w:
        left_hist = np.zeros(256, dtype=np.int64)

        for i in range(w - 1):

            column = sub[:, i]
            left_hist += np.bincount(column, minlength=256)

            if i + 1 < min_w or w - (i + 1) < min_w:
                continue

            right_hist = total_hist - left_hist

            mi = mutual_information_from_histograms(left_hist, right_hist)

            if mi > best_mi:
                best_mi = mi
                best_split = ("v", x0 + i + 1)

    # horizontal splits
    if h >= 2 * min_h:
        top_hist = np.zeros(256, dtype=np.int64)

        for i in range(h - 1):

            row = sub[i]
            top_hist += np.bincount(row, minlength=256)

            if i + 1 < min_h or h - (i + 1) < min_h:
                continue

            bottom_hist = total_hist - top_hist

            mi = mutual_information_from_histograms(top_hist, bottom_hist)

            if mi > best_mi:
                best_mi = mi
                best_split = ("h", y0 + i + 1)

    return best_split


def partition_heightmap(
    heightmap,
    max_depth=4,
    min_rel_size=0.1,
):
    """
    Recursively partitions a heightmap using mutual information splits.

    Returns list of leaf rectangles:
    (y0, y1, x0, x1)
    """

    leaves = []

    def recurse(y0, y1, x0, x1, depth):

        if depth >= max_depth:
            leaves.append((y0, y1, x0, x1))
            return

        split = best_split(heightmap, y0, y1, x0, x1, min_rel_size)

        if split is None:
            leaves.append((y0, y1, x0, x1))
            return

        axis, pos = split

        if axis == "v":
            recurse(y0, y1, x0, pos, depth + 1)
            recurse(y0, y1, pos, x1, depth + 1)
        else:
            recurse(y0, pos, x0, x1, depth + 1)
            recurse(pos, y1, x0, x1, depth + 1)

    h, w = heightmap.shape

    recurse(0, h, 0, w, 0)

    return leaves


def evaluate_composite_aesthetics_measure(
    heightmap: np.ndarray,
    compressor: callable,
    division_depth: int,
    reshape: bool = True,
):
    copy_heightmap = quantize_heightmap(heightmap)

    heightmap_division = partition_heightmap(copy_heightmap, division_depth, 0.2)

    def compute_ncd(sub1: np.ndarray, sub2: np.ndarray, compressor: callable):
        c_1 = compressor(sub1)
        c_2 = compressor(sub2)
        if reshape:
            sub12 = concatenate_domains_rect(sub1, sub2)
            if sub12 is None:
                return None
        else:
            sub12 = np.concatenate([sub1.reshape(1, -1), sub2.reshape(1, -1)])
        # print(sub1.shape, sub2.shape, sub12.shape)
        c_joint = compressor(sub12)

        return (c_joint - min(c_1, c_2)) / max(c_1, c_2)

    # TODO: plot the division onto the image
    ncds = []
    for i in range(len(heightmap_division)):
        y0, y1, x0, x1 = heightmap_division[i]
        sub_1 = copy_heightmap[y0:y1, x0:x1]
        for j in range(i, len(heightmap_division)):
            y0, y1, x0, x1 = heightmap_division[j]
            sub_2 = copy_heightmap[y0:y1, x0:x1]
            ncd = compute_ncd(sub_1, sub_2, compressor)
            print(ncd)
            if ncd is not None:
                ncds.append(ncd)
            else:
                print("Uncomputable NCD detected!")

    return 1.0 - np.mean(ncds)


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


def process_heightmap(
    heightmap: np.ndarray, filename: str, chunk_size: int, division_depth: int
):
    print("-" * 32)
    print(f"Processing heightmap {filename}")
    print("Shape:", heightmap.shape)
    print("Min:", heightmap.min(), "Max:", heightmap.max())

    erosion_score = evaluate_erosion_score(heightmap)
    print(f"Erosion score: {erosion_score}")

    gradient_score = evaluate_gradient_score(heightmap, chunk_size)
    print(f"Gradient score: {gradient_score}")

    zurek_png, zurek_jpg, zurek_zlib = evaluate_global_aesthetic_measure(heightmap)
    print(
        f"GAM:\n  png : ({zurek_png}) \n  jpg : ({zurek_jpg}) \n  zlib : ({zurek_zlib})"
    )

    order_png = evaluate_composite_aesthetics_measure(
        heightmap, compressed_size_png, division_depth
    )

    order_jpg = evaluate_composite_aesthetics_measure(
        heightmap, compressed_size_jpg, division_depth
    )

    order_zlib = evaluate_composite_aesthetics_measure(
        heightmap, compressed_size_zlib, division_depth
    )

    print(
        f"CAM:\n  png : ({order_png}) \n  jpg : ({order_jpg}) \n  zlib : ({order_zlib})"
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
            heightmap = read_exr_grayscale(path)
        elif fmt in ("jpg", "jpeg"):
            heightmap = read_jpg_grayscale(path)
        elif fmt in ("tif", "tiff"):
            heightmap = read_tiff_grayscale(path)
        else:
            raise RuntimeError("Unexpected format")

        process_heightmap(heightmap, filename, args.chunk_size, args.division_depth)

        del heightmap


if __name__ == "__main__":
    main()
