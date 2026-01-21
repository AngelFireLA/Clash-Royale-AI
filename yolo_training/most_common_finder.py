#!/usr/bin/env python3
import os
from PIL import Image

# Allowed resolutions
ALLOWED_RESOLUTIONS = {(562, 820), (562, 806)}

# Folders (adjust these if your train folder lives elsewhere)
TRAIN_DIR = "dataset/test"
IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
LABEL_DIR = os.path.join(TRAIN_DIR, "label")

# Supported image file extensions.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")

def process_images():
    """
    Processes all images in the IMAGES_DIR folder:
      - Keeps images with an allowed resolution (562x820 or 562x806) and records their names.
      - Deletes any image that is not of an allowed resolution.
      - For every deleted image, if there is a corresponding label in LABEL_DIR (same file name),
        the label is also deleted.
    Returns:
        kept_images (list of tuples): Each tuple is (filename, resolution) for kept images.
        total_images (int): Total number of images processed.
    """
    kept_images = []  # List of tuples: (filename, resolution)
    total_images = 0

    for filename in os.listdir(IMAGES_DIR):
        # Process only images with valid extensions.
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            total_images += 1
            image_path = os.path.join(IMAGES_DIR, filename)
            try:
                with Image.open(image_path) as img:
                    resolution = img.size  # (width, height)
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
                continue

            if resolution in ALLOWED_RESOLUTIONS:
                kept_images.append((filename, resolution))
            else:
                # Delete the image as its resolution is not allowed
                try:
                    os.remove(image_path)
                    print(f"Deleted image: {filename} (resolution: {resolution[0]}x{resolution[1]})")
                except Exception as e:
                    print(f"Error deleting image {filename}: {e}")

                # Also attempt to delete its corresponding label
                label_path = os.path.join(LABEL_DIR, filename)
                if os.path.exists(label_path):
                    try:
                        os.remove(label_path)
                        print(f"Deleted label for image: {filename}")
                    except Exception as e:
                        print(f"Error deleting label {filename}: {e}")
                else:
                    print(f"No label found for image {filename}")

    return kept_images, total_images

def print_summary(kept_images, total_images):
    """
    Prints a summary that groups the kept images by resolution and displays,
    for each allowed resolution, the number of images and the percentage of total images.
    It also lists the image names.
    """
    if total_images == 0:
        print("No images were processed.")
        return

    # Group kept images by resolution.
    resolution_dict = {}
    for filename, resolution in kept_images:
        resolution_dict.setdefault(resolution, []).append(filename)

    print("\nSummary of kept images:")
    print("Resolution        | Count | Percentage")
    print("-----------------------------------------")
    for res in sorted(resolution_dict.keys()):
        count = len(resolution_dict[res])
        percentage = (count / total_images) * 100
        # Format the resolution as "width x height"
        print(f"{res[0]}x{res[1]:<10} | {count:<5} | {percentage:.2f}%")
        print("Image names:")
        for name in resolution_dict[res]:
            print("  -", name)
        print()

    print(f"Total images processed: {total_images}")
    kept_count = sum(len(names) for names in resolution_dict.values())
    removed_count = total_images - kept_count
    print(f"Kept images: {kept_count}")
    print(f"Deleted images: {removed_count}")

def main():
    kept_images, total_images = process_images()
    print_summary(kept_images, total_images)

if __name__ == "__main__":
    main()
