import numpy as np
import OpenEXR
import Imath
import math
from PIL import Image
from scipy.stats import entropy
import tifffile
import zlib
import lzma
import io
import porespy as pspy
import matplotlib.pyplot as plt
import scipy as spy
import os
import tqdm

JPEG_MAX_DIM = 65500


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


def compressed_size_lzma(arr):
    return len(lzma.compress(arr.tobytes()))


# assumes the values are positive
def quantize_heightmap(
    heightmap: np.ndarray,
    max_value_of_population: float,
    desired_max_value: float = 255.0,
    type=np.uint8,
):
    return np.round(desired_max_value * (heightmap / max_value_of_population)).astype(
        type
    )


# assumes the values are positive
def normalize_heightmap(heightmap: np.ndarray, max_value_of_population: float):
    assert heightmap.min() >= 0.0
    return np.clip(heightmap / max_value_of_population, 0.0, 1.0)


def shannon_entropy(arr):
    values, counts = np.unique(arr, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)


# global-local metric that contributes to the final metric basing on how "eroded" the terrain is
def evaluate_erosion_score(heightmap: np.ndarray, nbins: int = 256):
    assert (
        heightmap.min() >= 0.0 and heightmap.max() <= 1.0
    ), f"Actual min: {heightmap.min()}. Actual max: {heightmap.max()}"

    min_mean_delta = 1.0e-6  # the actual delta depends on ULPs of the float representation in python, but I stick to a fixed value
    max_std = 1.0  # for normalized heightmaps, the deviation of a single texel from the mean can equal at most 1.0
    max_score = max_std / min_mean_delta

    height, width = heightmap.shape

    deltas = []

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
            deltas.append(max_delta)

    mean = np.mean(deltas)
    variance = np.var(deltas)
    std_dev = math.sqrt(variance)

    erosion_score = (
        max(min_mean_delta, std_dev / mean if mean != 0 else 0.0) / max_score
    )

    assert erosion_score <= 1.0 and erosion_score >= 0.0

    maxEntropy = -(1.0 / nbins) * np.log2(1.0 / nbins) * nbins
    delta_entropy = 1.0 - (
        shannon_entropy(quantize_heightmap(np.array(deltas), 1.0, nbins - 1, np.uint8))
        / maxEntropy
    )

    assert (
        delta_entropy >= 0.0 and delta_entropy <= 1.0
    ), f"Actual delta entropy: {delta_entropy}"

    return erosion_score * delta_entropy


# TODO: think how the magnitude can be included
def evaluate_gradient_score(heightmap: np.ndarray, subdomainSize: int, nbins: int = 32):
    assert heightmap.min() >= 0.0 and heightmap.max() <= 1.0
    min_mean_gradient_span = 1.0e-6
    max_gradient_std = math.sqrt(8)
    max_gradient_score = max_gradient_std / min_mean_gradient_span

    gy, gx = np.gradient(heightmap)
    height, width = heightmap.shape

    domains_y = height // subdomainSize
    domains_x = width // subdomainSize
    num_domains = domains_y * domains_x

    average_gradients = np.empty((num_domains, 2))
    spans = np.empty(num_domains)

    # computes the average angular span of gradients per subdomain
    for y in range(domains_y):
        for x in range(domains_x):

            start_x = x * subdomainSize
            start_y = y * subdomainSize

            end_x = start_x + subdomainSize
            end_y = start_y + subdomainSize

            gx_local = gx[start_y:end_y, start_x:end_x]
            gy_local = gy[start_y:end_y, start_x:end_x]

            norm = np.sqrt(gx_local**2 + gy_local**2) + 1e-12

            g = np.vstack((gx_local / norm, gy_local / norm))
            dot_matrix = g @ g.T
            min_dot = np.clip(np.min(dot_matrix), -1.0, 1.0)
            max_dot_angle = np.abs((min_dot - 1.0) / 2.0)

            linear_idx = y * domains_x + x

            spans[linear_idx] = max_dot_angle

            avg_gradient = [
                np.mean(g[0]),
                np.mean(g[1]),
            ]
            avg_gradient /= np.linalg.norm(avg_gradient) + 1e-12
            average_gradients[linear_idx] = avg_gradient

    average_angular_span = np.mean(spans)
    angle_std = np.linalg.norm(np.std(average_gradients, axis=0))

    gradient_score = (
        angle_std / max(min_mean_gradient_span, average_angular_span)
    ) / max_gradient_score

    assert gradient_score >= 0.0 and gradient_score <= 1.0

    maxEntropy = -(1.0 / nbins) * np.log2(1.0 / nbins) * nbins

    gradients_as_angles = (
        np.angle([np.complex64(x, y) for (x, y) in average_gradients], True) + 180
    )
    gradients_entropy = 1.0 - (
        shannon_entropy(
            quantize_heightmap(gradients_as_angles, 360.0, nbins - 1, np.uint8)
        )
        / maxEntropy
    )

    assert (
        gradients_entropy >= 0.0 and gradients_entropy <= 1.0
    ), f"Actual gradient entropy: {gradients_entropy}"

    return gradient_score * gradients_entropy


def find_balanced_threshold(
    heightmap: np.ndarray, max_iter: int = 20, bounds=(0.47, 0.53)
):
    low, high = bounds

    best_threshold = 0.5
    best_error = float("inf")

    total = heightmap.size

    for _ in range(max_iter):
        mid = 0.5 * (low + high)

        binary = heightmap >= mid
        white = np.count_nonzero(binary)
        ratio = white / total

        error = abs(ratio - 0.5)

        if error < best_error:
            best_error = error
            best_threshold = mid

        if ratio > 0.5:
            low = mid
        else:
            high = mid

    return best_threshold


def evaluate_fractal_score(heightmap: np.ndarray, threshold: float = None):
    quantized_heightmap = normalize_heightmap(heightmap, np.max(heightmap))

    if threshold is None:
        threshold = find_balanced_threshold(quantized_heightmap)

    binary_heightmap = quantized_heightmap >= threshold

    # Image.fromarray(quantized_heightmap >= threshold).save(
    #     f"./{threshold}_{np.mean(heightmap)}_mask.png", format="PNG"
    # )

    # --- fractal dimension ---
    data = pspy.metrics.boxcount(binary_heightmap, 15)
    counts = np.array(data.count)
    sizes = np.array(data.size)
    non_zero_counts = np.where(counts > 0)[0]
    if len(non_zero_counts) != 0:
        coeffs = np.polyfit(
            np.log(sizes[non_zero_counts]), np.log(counts[non_zero_counts]), 1
        )
        fractal_dimension = (
            2.0
            if (np.isnan(coeffs[0]) or np.isinf(coeffs[0]))
            else -coeffs[
                0
            ]  # makes the dimension 2 by default -> as if we have a plain square
        )
    else:
        fractal_dimension = 2.0

    # --- power spectrum ---
    F = np.fft.fft2(heightmap - np.mean(heightmap))
    psd2D = np.abs(np.fft.fftshift(F)) ** 2

    ny, nx = psd2D.shape
    y, x = np.indices((ny, nx))
    center = np.array([(ny - 1) / 2, (nx - 1) / 2])

    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())

    radial_psd = tbin / nr
    freqs = np.arange(len(radial_psd))

    freqs, psd = freqs[1:], radial_psd[1:]

    slope, residuals = np.polyfit(np.log(freqs), np.log(psd), deg=1, full=True)[0]

    beta = 0.0 if (np.isnan(slope) or np.isinf(slope)) else -slope
    mse = residuals / len(freqs)

    return fractal_dimension, beta, mse


def evaluate_global_aesthetic_measure(quantized_heightmap: np.ndarray, compressors):
    height, width = quantized_heightmap.shape

    texel_entropy = shannon_entropy(quantized_heightmap)
    initial_information_content = (height * width) * texel_entropy

    global_measures = []

    for compressor in compressors:
        heightmap_compressed = compressor(quantized_heightmap.reshape(1, -1))
        measure = (
            initial_information_content - heightmap_compressed
        ) / initial_information_content
        global_measures.append(
            1.0 if (np.isneginf(measure) or np.isinf(measure)) else measure
        )

    return global_measures


def mutual_information_from_histograms(h1, h2):
    """
    Mutual information between two discrete distributions defined by histograms.
    """
    n1 = h1.sum()
    n2 = h2.sum()

    if n1 == 0 or n2 == 0:
        return 0.0

    # probabilities
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
    quantized_heightmap: np.ndarray,
    population_max: int,
    division_depth: int,
    compressors,
):
    # heightmap_division = partition_heightmap(quantized_heightmap, division_depth, 0.2)
    height, width = quantized_heightmap.shape
    heightmap_division = []
    h3 = int(height / division_depth)
    w3 = int(width / division_depth)
    for gx in range(division_depth):
        for gy in range(division_depth):
            heightmap_division.append((gx * h3, (gx + 1) * h3, gy * w3, (gy + 1) * w3))

    split_visualization_heightmap = quantized_heightmap.copy()

    def compute_ncd(sub1, sub2, compressor):
        flat1 = sub1.ravel()
        flat2 = sub2.ravel()

        c1 = compressor(flat1)
        c2 = compressor(flat2)

        cj = compressor(np.hstack((flat1, flat2)))

        return (cj - min(c1, c2)) / max(c1, c2)

    ncds = {c: [] for c in compressors}
    for i in range(len(heightmap_division)):
        y0, y1, x0, x1 = heightmap_division[i]
        sub_1 = quantized_heightmap[y0:y1, x0:x1]

        split_visualization_heightmap[y0:y1, x0] = population_max
        split_visualization_heightmap[y0:y1, min(x1, width - 1)] = population_max
        split_visualization_heightmap[y0, x0:x1] = population_max
        split_visualization_heightmap[min(y1, height - 1), x0:x1] = population_max

        for j in range(i + 1, len(heightmap_division)):
            y0, y1, x0, x1 = heightmap_division[j]
            sub_2 = quantized_heightmap[y0:y1, x0:x1]
            for compressor in compressors:
                ncd = compute_ncd(sub_1, sub_2, compressor)
                ncds[compressor].append(ncd)
    cams = []
    for compressor in compressors:
        cams.append(1.0 - np.mean(ncds[compressor]))

    return (cams, split_visualization_heightmap)


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


def get_metric_vector(
    heightmap: np.ndarray,
    chunk_size: int,
    division_depth: int,
    population_max: float,
    verbose: bool = False,
):
    quantized_heightmap = quantize_heightmap(heightmap, population_max, 255.0)
    normalized_heightmap = normalize_heightmap(heightmap, population_max)

    erosion_score = evaluate_erosion_score(normalized_heightmap)
    gradient_score = evaluate_gradient_score(normalized_heightmap, chunk_size)

    fractal_dimension, beta, mse = evaluate_fractal_score(heightmap, 0.5)

    zurek_lzma = evaluate_global_aesthetic_measure(
        quantized_heightmap, [compressed_size_lzma]
    )[0]

    (order_png, order_lzma), split_visualization = (
        evaluate_composite_aesthetics_measure(
            quantized_heightmap,
            population_max,
            division_depth,
            [
                compressed_size_png,
                compressed_size_lzma,
            ],
        )
    )

    # Image.fromarray(split_visualization).save(
    #     f"./{np.mean(heightmap):.3f}_split.png", format="PNG"
    # )

    if verbose:
        print("-" * 32)
        print(f"Erosion score: {erosion_score}")
        print(f"Gradient score: {gradient_score}")
        print(f"Fractal dimenison: {fractal_dimension}")
        print(f"Noise beta exponent: {beta}. Fit MSE: {mse}")
        print(f"GAM:\n  lzma : ({zurek_lzma})")
        print(f"CAM:\n  png : ({order_png}) \n  lzma : ({order_lzma})")
        print("+" * 32)

    return np.array(
        [
            erosion_score,
            gradient_score,
            fractal_dimension,
            beta,
            zurek_lzma,
            # order_png,
            order_lzma,
        ],
        dtype=np.float64,
    )


def build_metric_vectors(
    directory: str,
    chunk_size: int,
    division_depth: int,
    fmt: str,
    data_fraction: float = 1.0,
    if_quad_data: bool = False,
):
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")

    extensions = {
        "exr": ([".exr"], read_exr_grayscale, 1.0),
        "jpg": ([".jpg", ".jpeg"], read_jpg_grayscale, 255.0),
        "jpeg": ([".jpg", ".jpeg"], read_jpg_grayscale, 255.0),
        # "tif": ([".tif", ".tiff"],  read_tiff_grayscale),
        # "tiff": ([".tif", ".tiff"], read_tiff_grayscale),
    }

    if fmt not in extensions:
        raise ValueError("Unsupported format")

    possible_extensions, file_reader, population_max = extensions[fmt]

    files = np.array(
        sorted(
            f
            for f in os.listdir(directory)
            if any(f.lower().strip().endswith(ext) for ext in possible_extensions)
        )
    )
    files = files[np.random.randint(0, len(files), int(data_fraction * len(files)))]

    print(f"Total heightmaps to evaluate: {len(files)}")

    metric_vectors = []

    if if_quad_data:
        for filename in tqdm.tqdm(files):
            path = os.path.join(directory, filename)
            heightmap = file_reader(path)
            half_side_len = int(len(heightmap) / 2)

            print(f"\nProcessing heightmap: {filename}")

            for sx in range(2):
                for sy in range(2):
                    subterrain = heightmap[
                        sx * half_side_len : (sx + 1) * half_side_len,
                        sy * half_side_len : (sy + 1) * half_side_len,
                    ]

                    height_max = subterrain.max()
                    if (
                        np.isnan(height_max)
                        or np.isnan(subterrain.min())
                        or np.isclose(height_max, 0.0)
                    ):
                        print("Skipping tile! Invalid values")
                        continue

                    if np.mean(subterrain > (population_max * 0.05)) <= 0.6:
                        print("Skipping tile! Too much water")
                        continue

                    print(f"Terrain split: {sx * 2 + sy}")
                    metric_vector = get_metric_vector(
                        subterrain, chunk_size, division_depth, subterrain.max(), True
                    )

                    metric_vectors.append(metric_vector)

            del heightmap
    else:
        for filename in tqdm.tqdm(files):
            path = os.path.join(directory, filename)
            heightmap = file_reader(path)

            print(f"\nProcessing heightmap: {filename}")

            if np.mean(heightmap > (population_max * 0.05)) <= 0.6:
                print("Skipping terrain! Too much water")
                continue

            metric_vector = get_metric_vector(
                heightmap, chunk_size, division_depth, population_max, True
            )

            metric_vectors.append(metric_vector)

            del heightmap

    return np.array(metric_vectors)
