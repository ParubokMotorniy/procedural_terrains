import numpy as np
import csv
import time
import argparse
import io
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Please install the SDK: pip install google-genai pillow")
    exit()

API_KEY = "AIzaSyC1Vn3SawCqPPOIewmxQRNulvOwwT2fKDg"
GEN_MODEL = "gemini-2.5-flash-image"
CLASS_MODEL = "gemini-2.5-flash-lite"

GEN_RPM = 250
CLASS_RPM = 150


def get_client():
    return genai.Client(api_key=API_KEY)


def generate_routine(
    num_images,
    output_dir,
    user_prompt="Generate a heightmap of terrain for drop-in use as a place of action in a videogame. The only artistic requirement: make it boring and unappealing from the perspective of a potential player while preserving the feaures, objects and structure that are typical of terrain",
):
    client = get_client()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_prompt = f"{user_prompt}. Technical limitations are: output MUST be a strictly 1024x1024 orthographic top-down grayscale heightmap. No colors, no shadows, no cloud obstruction, no actual water bodies. As if it was a real digital elevation model of total spatial resolution 100 kilometers per heightmap side."

    print(f"Starting generation of {num_images} images...")

    for i in tqdm(range(num_images)):
        filename = output_path / f"terrain_{i:04d}.jpg"

        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            # Extract image from response parts
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    img = Image.open(io.BytesIO(part.inline_data.data)).convert("L")
                    img.save(filename)
                    print(f"[{i+1}/{num_images}] Saved: {filename.name}")
                    break

            # Respect RPM Limit (60s / 10 requests = 6s delay)
            time.sleep(60 / GEN_RPM)

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(
                    f"\n[!] QUOTA REACHED or RATE LIMITED. Aborting to prevent lockout.\nError: {e}"
                )
                break
            else:
                print(f"\n[!] ERROR at image {i}: {e}")


def classification_routine(
    images_dir,
    csv_output,
    class_prompt="The provided image is a grayscale terrain heightmap. If you were the player of a videogame that uses this heightmap as a place of action, would you find this terrain INTERESTING and APPEALING? Rate this heightmap on a linear scale from 0 to 1, where 0 represents BORING and UNAPPEALING and 1 represents INTERESTING and APPEALING. Give answer as a single float value with up to three decimal places.",
):
    image_paths = [
        str(p)
        for p in Path(images_dir).glob("*")
        if p.suffix.strip().lower() in [".png", ".jpg", ".jpeg"]
    ]

    client = get_client()
    results = []
    raw_classification = []

    print(f"Classifying {len(image_paths)} images...")

    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            continue

        try:
            with open(path, "rb") as f:
                img_bytes = f.read()

            response = client.models.generate_content(
                model=CLASS_MODEL,
                contents=[
                    class_prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ],
            )

            result_text = float(response.text.strip())
            results.append([path.name, result_text])
            raw_classification.append(result_text)
            print(f"Classified {path.name}: {result_text}")

            time.sleep(60 / CLASS_RPM)

        except Exception as e:
            print(f"[!] Error processing {path.name}: {e}")
            results.append([path.name, "ERROR"])
            if "429" in str(e):
                break

    # Write to CSV
    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Classification"])
        writer.writerows(results)
    print(f"Results dumped to {csv_output}")

    return np.array(raw_classification, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini API Microservice Script")
    parser.add_argument(
        "--mode", choices=["gen", "class"], required=True, help="Toggle mode"
    )
    parser.add_argument(
        "--prompt", type=str, help="The prompt for generation or classification"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--dir", type=str, default="./output", help="Directory for storage/input"
    )
    parser.add_argument(
        "--csv", type=str, default="results.csv", help="CSV output for classification"
    )

    args = parser.parse_args()

    if args.mode == "gen":
        generate_routine(args.count, args.dir, args.prompt)
    elif args.mode == "class":
        classification_routine(args.dir, args.csv, args.prompt)
