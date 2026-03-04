import argparse
import os
import numpy as np
import OpenEXR
import Imath
import math
import scipy as spy


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
                    gx[
                        start_y:end_y,
                        start_x:end_x,
                    ],
                    gy[
                        start_y:end_y,
                        start_x:end_x,
                    ],
                ]
            )

            norm = np.linalg.norm(local_gradients, axis=0) + 1e-12

            normalized_gradients = local_gradients / norm

            max_dot_angle = 0
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

    max_global_angle = 0
    average_gradients = np.array(average_gradients)
    g = average_gradients.reshape(2, -1).T 
    dot_matrix = g @ g.T
    min_dot = np.clip(np.min(dot_matrix), -1.0, 1.0)
    max_global_angle = math.acos(min_dot)

    average_angle = np.mean(spans)
    gradient_score = max_global_angle / max(1e-12, average_angle)

    return gradient_score


def evaluate_fractal_score(heightmap: np.ndarray):
    pass


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


def process_heightmap(heightmap: np.ndarray, filename: str):
    print(f"Processing heightmap {filename}")
    print("Shape:", heightmap.shape)
    print("Min:", heightmap.min(), "Max:", heightmap.max())

    erosion_score = evaluate_erosion_score(heightmap)
    print(f"Erosion score: {erosion_score}")

    gradient_score = evaluate_gradient_score(heightmap, 8)
    print(f"Gradient score: {gradient_score}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for the heightmaps.")
    parser.add_argument(
        "directory", type=str, help="Directory containing EXR heightmaps"
    )
    args = parser.parse_args()

    directory = args.directory

    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory")

    exr_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".exr"))

    for filename in exr_files:
        path = os.path.join(directory, filename)

        heightmap = read_exr_grayscale(path)

        process_heightmap(heightmap, filename)

        del heightmap


if __name__ == "__main__":
    main()
