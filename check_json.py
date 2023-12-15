import json
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def check_image(image_info, image_root):
    image_path = os.path.join(image_root, image_info['image'])
    if not os.path.exists(image_path) or not is_image_valid(image_path):
        return image_info['image']
    return None

def find_missing_or_corrupt_images(json_file, image_root, output_file, max_workers=10):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    missing_or_corrupt_images = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda info: check_image(info, image_root), data)

    for result in results:
        if result is not None:
            missing_or_corrupt_images.append(result)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for image in missing_or_corrupt_images:
            out_file.write(image + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find missing or corrupt images from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("image_root", help="Root directory of the images")
    parser.add_argument("output_file", help="Path to the output text file for missing or corrupt images")

    args = parser.parse_args()

    find_missing_or_corrupt_images(args.json_file, args.image_root, args.output_file)
