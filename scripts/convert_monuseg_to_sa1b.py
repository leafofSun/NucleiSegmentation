#!/usr/bin/env python3
"""
Convert MoNuSeg dataset (Tissue Images + Annotations) to SA-1B style dataset.

Output layout (default):
  output_dir/
    images/
    masks/
    image2label_train.json   # image -> [mask,...]
    label2image_test.json    # mask -> image

Usage:
  python scripts/convert_monuseg_to_sa1b.py --input-root data/MoNuSeg --output-dir data/monuseg_sa1b

This script:
 - Reads .tif images from MoNuSegTrainingData/Tissue Images and corresponding .xml in Annotations
 - Rasterizes each Region (polygon) from XML into a binary PNG mask
 - Saves images as PNG and masks as 8-bit binary PNGs
 - Produces JSON mappings compatible with the repo examples
"""
import argparse
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Convert MoNuSeg to SA-1B style dataset")
    p.add_argument("--input-root", type=str, default="data/MoNuSeg",
                   help="Path to the MoNuSeg folder (contains MoNuSegTrainingData and MoNuSegTestData)")
    p.add_argument("--output-dir", type=str, default="data/monuseg_sa1b",
                   help="Directory to write SA-1B style dataset")
    p.add_argument("--processing-limit", type=int, default=0,
                   help="If >0, limit number of images processed (for quick tests)")
    p.add_argument("--force", action='store_true',
                   help="If set, force re-generation of images and masks (overwrite existing files)")
    return p.parse_args()


def xml_regions_to_polygons(xml_path):
    """Parse Regions from an Aperio-style XML and return list of polygons (list of (x,y) tuples)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polys = []
    # Regions are under //Region and Vertices/Vertex with X and Y
    for region in root.findall('.//Region'):
        verts = []
        verts_el = region.find('Vertices')
        if verts_el is None:
            continue
        for v in verts_el.findall('Vertex'):
            x = v.get('X')
            y = v.get('Y')
            try:
                xi = float(x)
                yi = float(y)
            except Exception:
                continue
            verts.append((xi, yi))
        if len(verts) >= 3:
            polys.append(verts)
    return polys


def rasterize_polygon(poly, size):
    """Rasterize a polygon (float coords) to a binary mask of given (width,height)."""
    W, H = size
    # PIL expects sequence of tuples in pixel coordinates; round to int
    pts = [(int(round(x)), int(round(y))) for (x, y) in poly]
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
    arr = np.array(mask, dtype=np.uint8) * 255
    return Image.fromarray(arr)


def convert_folder(training_images_dir, annotations_dir, out_images_dir, out_masks_dir, image2label, limit=0, force=False):
    training_images_dir = Path(training_images_dir)
    annotations_dir = Path(annotations_dir)
    out_images_dir = Path(out_images_dir)
    out_masks_dir = Path(out_masks_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in training_images_dir.iterdir() if p.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']])
    if limit and limit > 0:
        imgs = imgs[:limit]
    for img_p in imgs:
        base = img_p.stem
        xml_p = annotations_dir / (base + '.xml')
        if not xml_p.exists():
            # try other common suffixes
            xml_alt = annotations_dir / (base + '.xml')
            if not xml_alt.exists():
                print(f"Warning: annotation for {img_p.name} not found, skip")
                continue
            xml_p = xml_alt

        # load image to know size and save as PNG (unless present and not forcing)
        out_img_rel = os.path.join('images', base + '.png').replace('\\', '/')
        out_img_path = out_images_dir / (base + '.png')
        with Image.open(img_p) as im:
            im = im.convert('RGB')
            W, H = im.size
            if not out_img_path.exists() or force:
                im.save(out_img_path)

        polys = xml_regions_to_polygons(xml_p)
        mask_paths = []
        for i, poly in enumerate(polys):
            mask_name = f"{base}_mask_{i:04d}.png"
            out_mask_path = out_masks_dir / mask_name
            # create mask if missing or forcing
            if not out_mask_path.exists() or force:
                mask_img = rasterize_polygon(poly, (W, H))
                mask_img.save(out_mask_path)
            mask_paths.append(os.path.join('masks', mask_name).replace('\\', '/'))

        image2label[out_img_rel] = mask_paths


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    out_root = Path(args.output_dir)
    out_images_dir = out_root / 'images'
    out_masks_dir = out_root / 'masks'

    image2label = {}
    label2image = {}

    # Process training data
    train_images_dir = input_root / 'MoNuSegTrainingData' / 'Tissue Images'
    train_ann_dir = input_root / 'MoNuSegTrainingData' / 'Annotations'
    if train_images_dir.exists() and train_ann_dir.exists():
        print('Converting training images...')
        convert_folder(train_images_dir, train_ann_dir, out_images_dir, out_masks_dir, image2label, limit=args.processing_limit, force=args.force)
    else:
        print('Training data not found in expected location:', train_images_dir)

    # Process test data (if present) - will append mappings too
    test_images_dir = input_root / 'MoNuSegTestData' / 'Tissue Images'
    test_ann_dir = input_root / 'MoNuSegTestData' / 'Annotations'
    if test_images_dir.exists() and test_ann_dir.exists():
        print('Converting test images...')
        convert_folder(test_images_dir, test_ann_dir, out_images_dir, out_masks_dir, image2label, limit=args.processing_limit, force=args.force)

    # Build reverse mapping mask -> image
    for img, masks in image2label.items():
        for m in masks:
            label2image[m] = img

    out_root.mkdir(parents=True, exist_ok=True)
    train_json_path = out_root / 'image2label_train.json'
    test_json_path = out_root / 'label2image_test.json'

    # If JSONs exist, merge instead of overwriting: union masks per image
    if train_json_path.exists():
        try:
            with open(train_json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        existing = {}

    # merge existing and new image2label: union lists
    merged_image2label = dict(existing)
    for img, masks in image2label.items():
        if img in merged_image2label:
            # union preserve order: existing first, then new unique
            exist_masks = merged_image2label[img]
            combined = list(exist_masks)
            for m in masks:
                if m not in combined:
                    combined.append(m)
            merged_image2label[img] = combined
        else:
            merged_image2label[img] = masks

    # build label2image from merged mapping (and keep any existing label2image mapping for masks not in merged)
    if test_json_path.exists():
        try:
            with open(test_json_path, 'r', encoding='utf-8') as f:
                existing_label2image = json.load(f)
        except Exception:
            existing_label2image = {}
    else:
        existing_label2image = {}

    merged_label2image = dict(existing_label2image)
    for img, masks in merged_image2label.items():
        for m in masks:
            merged_label2image[m] = img

    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_image2label, f, indent=4, ensure_ascii=False)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_label2image, f, indent=4, ensure_ascii=False)

    print('Done. Wrote:', train_json_path, test_json_path)


if __name__ == '__main__':
    main()
