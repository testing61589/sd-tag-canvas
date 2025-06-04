import os
import shutil
import sys
import argparse
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


HASH_THRESHOLD = 1  # Allow minor differences in image content
DUPLICATE_FOLDER = "duplicates"
ANIMATED_FOLDER = "animated"
VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mov', '.avi', '.mkv')

import concurrent.futures

def _tag_greyscale_image_worker(path, folder_path):
    """
    Worker that returns a result for the main thread to handle.
    """
    base_name, _ = os.path.splitext(os.path.basename(path))
    txt_path = os.path.join(folder_path, base_name + '.txt')

    # Skip if already tagged
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            if 'greyscale' in f.read().lower():
                return None

    if is_greyscale_image(path):
        return txt_path  # Tell main process to tag it
    return None

def check_and_tag_greyscale_images(folder_path):
    abs_folder = os.path.abspath(folder_path)
    image_paths = [
        os.path.join(abs_folder, name)
        for name in os.listdir(abs_folder)
        if os.path.isfile(os.path.join(abs_folder, name)) and
           name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'))
    ]

    modified = 0
    created = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_tag_greyscale_image_worker, path, abs_folder): path for path in image_paths}

        with tqdm(total=len(futures), desc="Scanning for greyscale images", unit="img") as pbar:
            for future in as_completed(futures):
                txt_path = future.result()
                if txt_path:
                    if os.path.exists(txt_path):
                        with open(txt_path, 'a', encoding='utf-8') as f:
                            f.write(', greyscale\n')
                        tqdm.write(f"📝 Appended ', greyscale' to {os.path.basename(txt_path)}")
                        modified += 1
                    else:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write('greyscale\n')
                        tqdm.write(f"📝 Created {os.path.basename(txt_path)} with 'greyscale'")
                        created += 1
                pbar.update(1)

    print("\n📊 Greyscale tagging complete:")
    print(f"   ✍️  Modified existing .txt files: {modified}")
    print(f"   📄 Created new .txt files:       {created}")
    print(f"   📁 Total processed images:       {len(futures)}")


def is_animated_gif(filepath):
    try:
        with Image.open(filepath) as img:
            return getattr(img, "is_animated", False) and img.n_frames > 1
    except Exception:
        return False

def move_animated_and_video(folder_path):
    abs_folder = os.path.abspath(folder_path)
    animated_dir = os.path.join(abs_folder, ANIMATED_FOLDER)
    os.makedirs(animated_dir, exist_ok=True)

    for name in os.listdir(abs_folder):
        path = os.path.join(abs_folder, name)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1].lower()

        if ext in VIDEO_EXTENSIONS:
            print(f"🎞 Moving video: {os.path.relpath(path, abs_folder)}")
            shutil.move(path, os.path.join(animated_dir, name))

        elif ext == ".gif" and is_animated_gif(path):
            print(f"🌀 Moving animated GIF: {os.path.relpath(path, abs_folder)}")
            shutil.move(path, os.path.join(animated_dir, name))

            # Move accompanying .txt file if it exists
            base_name, _ = os.path.splitext(name)
            txt_path = os.path.join(abs_folder, base_name + '.txt')
            if os.path.exists(txt_path):
                print(f"📄 Moving .txt: {os.path.relpath(txt_path, abs_folder)}")
                shutil.move(txt_path, os.path.join(animated_dir, os.path.basename(txt_path)))



def get_image_info(filepath):
    with Image.open(filepath) as img:
        width, height = img.size
    file_size = os.path.getsize(filepath)
    return width, height, file_size

def compute_hash(filepath):
    with Image.open(filepath) as img:
        return imagehash.phash(img)

def group_similar_images(folder_path):
    hash_map = defaultdict(list)

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if not os.path.isfile(path):
            continue

        if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')):
            continue

        try:
            img_hash = compute_hash(path)

            matched = False
            for existing_hash in hash_map:
                if img_hash - existing_hash <= HASH_THRESHOLD:
                    hash_map[existing_hash].append(path)
                    matched = True
                    break

            if not matched:
                hash_map[img_hash].append(path)
        except Exception as e:
            print(f"Error processing {name}: {e}")

    return hash_map


def select_highest_quality(paths):
    def score(p):
        width, height, size = get_image_info(p)
        ext = os.path.splitext(p)[1].lower()
        is_png = ext == '.png'
        return (width * height, is_png, size)  # Higher is better for all fields

    return max(paths, key=score)

def is_greyscale_image(filepath, sample_ratio=1.0, grey_tolerance=30, bw_tolerance=30, threshold=0.90):
    """
    Detects if an image is greyscale, line art, or high-contrast B&W.
    - grey_tolerance: max channel difference to consider a pixel greyscale
    - bw_tolerance: range near 0 or 255 to consider as black/white
    - threshold: required proportion of grey/BW pixels
    """
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')
            pixels = list(img.getdata())

            sample_size = int(len(pixels) * sample_ratio)
            sampled = pixels[::max(1, len(pixels) // sample_size)]

            def is_grey(r, g, b):
                return max(r, g, b) - min(r, g, b) <= grey_tolerance

            def is_bw(r, g, b):
                return (
                    max(r, g, b) <= bw_tolerance or           # near black
                    min(r, g, b) >= 255 - bw_tolerance        # near white
                )

            good_pixels = sum(
                is_grey(r, g, b) or is_bw(r, g, b)
                for r, g, b in sampled
            )

            return (good_pixels / len(sampled)) >= threshold
    except Exception as e:
        print(f"⚠️ Error checking greyscale for {filepath}: {e}")
        return False



def move_duplicates(folder_path):
    abs_folder = os.path.abspath(folder_path)
    duplicate_dir = os.path.join(abs_folder, DUPLICATE_FOLDER)
    os.makedirs(duplicate_dir, exist_ok=True)

    hash_groups = group_similar_images(abs_folder)

    for group in hash_groups.values():
        if len(group) > 1:
            best = select_highest_quality(group)
            print(f"✔ Keeping: {os.path.relpath(best, abs_folder)}")

            for img in group:
                if img != best:
                    base_name, _ = os.path.splitext(os.path.basename(img))
                    txt_path = os.path.join(os.path.dirname(img), base_name + '.txt')

                    print(f"📦 Moving: {os.path.relpath(img, abs_folder)}")
                    shutil.move(img, os.path.join(duplicate_dir, os.path.basename(img)))

                    if os.path.exists(txt_path):
                        print(f"📦 Moving associated text: {os.path.basename(txt_path)}")
                        shutil.move(txt_path, os.path.join(duplicate_dir, os.path.basename(txt_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize and de-duplicate images, including greyscale detection.")
    parser.add_argument("folder", help="Path to folder containing images (relative or absolute)")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"❌ Error: Folder '{args.folder}' does not exist.")
        sys.exit(1)

    move_animated_and_video(args.folder)       # Step 1: Remove animated and video files
    move_duplicates(args.folder)               # Step 2: Deduplicate and move lower-quality duplicates
    # check_and_tag_greyscale_images(args.folder)  # Step 3: Tag greyscale on remaining top-level images
