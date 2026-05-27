import os
import json
import glob
import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from skimage import measure


try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.spatial import KDTree as ScipyKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False


# ==============================================================================
# 1. Organ knowledge
# ==============================================================================
DEFAULT_ORGAN_KNOWLEDGE = {
    "Adrenal_gland": {"context": "adrenal gland tissue", "desc": "adrenocortical cells"},
    "Bile-duct": {"context": "biliary tissue", "desc": "cholangiocytes"},
    "Bladder": {"context": "urothelial tissue", "desc": "transitional epithelial cells"},
    "Breast": {"context": "mammary tissue", "desc": "ductal epithelial cells"},
    "Cervix": {"context": "cervical tissue", "desc": "squamous epithelial cells"},
    "Colon": {"context": "colonic mucosa", "desc": "columnar epithelial cells"},
    "Esophagus": {"context": "esophageal tissue", "desc": "squamous cells"},
    "HeadNeck": {"context": "head and neck tissue", "desc": "squamous epithelial cells"},
    "Kidney": {"context": "renal tissue", "desc": "tubular epithelial cells"},
    "Liver": {"context": "hepatic tissue", "desc": "hepatocytes"},
    "Lung": {"context": "pulmonary tissue", "desc": "pneumocytes and macrophages"},
    "Ovarian": {"context": "ovarian tissue", "desc": "stromal and epithelial cells"},
    "Pancreatic": {"context": "pancreatic tissue", "desc": "acinar cells"},
    "Prostate": {"context": "prostatic tissue", "desc": "glandular epithelial cells"},
    "Skin": {"context": "cutaneous tissue", "desc": "keratinocytes"},
    "Stomach": {"context": "gastric mucosa", "desc": "glandular cells"},
    "Testis": {"context": "testicular tissue", "desc": "germ cells"},
    "Thyroid": {"context": "thyroid tissue", "desc": "follicular cells"},
    "Uterus": {"context": "uterine tissue", "desc": "endometrial cells"},
    "Brain": {"context": "brain tissue", "desc": "glial cells and neurons"},
    "Generic": {"context": "histopathology tissue", "desc": "cell nuclei"},
}

ORGAN_TO_ID = {
    "Adrenal_gland": 0,
    "Bile-duct": 1,
    "Bladder": 2,
    "Breast": 3,
    "Cervix": 4,
    "Colon": 5,
    "Esophagus": 6,
    "HeadNeck": 7,
    "Kidney": 8,
    "Liver": 9,
    "Lung": 10,
    "Ovarian": 11,
    "Pancreatic": 12,
    "Prostate": 13,
    "Skin": 14,
    "Stomach": 15,
    "Testis": 16,
    "Thyroid": 17,
    "Uterus": 18,
    "Brain": 19,
    "Generic": 20,
}


ATTRIBUTE_ORDER = ["color", "shape", "arrangement", "size", "density"]

LABEL_SPACES = {
    "color": ["deep-purple stained", "light-purple stained"],
    "shape": ["round", "oval", "elongated"],
    "arrangement": ["uniformly arranged", "disordered/clustered"],
    "size": ["small-sized", "medium-sized", "large-sized"],
    "density": ["sparsely distributed", "moderately distributed", "densely distributed"],
}

DEFAULT_ATTR_LABELS = [0, 0, 0, 1, 1]


# ==============================================================================
# 2. General utilities
# ==============================================================================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def normalize_text(x: Any) -> str:
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def normalize_organ_name(raw_organ: Any) -> str:
    if raw_organ is None:
        return "Generic"

    raw = str(raw_organ).strip()
    if not raw:
        return "Generic"

    raw_low = raw.lower().replace("_", "").replace("-", "").replace(" ", "")

    for organ in DEFAULT_ORGAN_KNOWLEDGE.keys():
        organ_low = organ.lower().replace("_", "").replace("-", "").replace(" ", "")
        if organ_low == raw_low:
            return organ

    for organ in DEFAULT_ORGAN_KNOWLEDGE.keys():
        organ_low = organ.lower().replace("_", "").replace("-", "").replace(" ", "")
        if organ_low in raw_low or raw_low in organ_low:
            return organ

    return "Generic"


def infer_split(rel_path: str, json_data: Any) -> str:
    if isinstance(json_data, dict):
        for key in ["split", "mode", "set"]:
            if key in json_data:
                value = str(json_data[key]).lower().strip()
                if value in {"train", "val", "valid", "validation", "test"}:
                    if value in {"valid", "validation"}:
                        return "val"
                    return value

    parts = rel_path.replace("\\", "/").lower().split("/")
    if "train" in parts:
        return "train"
    if "val" in parts:
        return "val"
    if "valid" in parts:
        return "val"
    if "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"

    return "train"


def find_image_path(json_path: str) -> Optional[str]:
    stem = os.path.splitext(json_path)[0]
    candidates = [
        stem + ".png",
        stem + ".jpg",
        stem + ".jpeg",
        stem + ".tif",
        stem + ".tiff",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def load_json_any(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_record_from_json(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        if len(data) == 0:
            return {}
        if isinstance(data[0], dict) and "annotations" in data[0]:
            return data[0]

    return {}


def get_image_hw(record: Dict[str, Any], image: Optional[np.ndarray]) -> Tuple[int, int]:
    if image is not None:
        h, w = image.shape[:2]
        return int(h), int(w)

    if "image" in record and isinstance(record["image"], dict):
        h = record["image"].get("height", None)
        w = record["image"].get("width", None)
        if h is not None and w is not None:
            return int(h), int(w)

    h = record.get("height", None)
    w = record.get("width", None)
    if h is not None and w is not None:
        return int(h), int(w)

    annotations = record.get("annotations", [])
    for ann in annotations:
        seg = ann.get("segmentation", None)
        if isinstance(seg, dict) and "size" in seg:
            size = seg["size"]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return int(size[0]), int(size[1])

    return 256, 256


def extract_organ_from_record(record: Dict[str, Any]) -> str:
    candidate_keys = [
        "organ_id",
        "organ",
        "organ_type",
        "tissue",
        "tissue_type",
        "site",
    ]

    for key in candidate_keys:
        if key in record:
            organ = normalize_organ_name(record.get(key))
            if organ != "Generic":
                return organ

    if "image" in record and isinstance(record["image"], dict):
        for key in candidate_keys:
            if key in record["image"]:
                organ = normalize_organ_name(record["image"].get(key))
                if organ != "Generic":
                    return organ

    return "Generic"


# ==============================================================================
# 3. Mask decoding
# ==============================================================================
def decode_segmentation_to_mask(seg: Any, h: int, w: int, inst_id: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    if seg is None:
        return mask

    # Polygon format.
    if isinstance(seg, list):
        if len(seg) == 0:
            return mask

        # COCO polygon can be either [flat_poly] or flat_poly.
        if all(isinstance(x, (int, float)) for x in seg):
            polygons = [seg]
        else:
            polygons = seg

        for poly in polygons:
            try:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] >= 3:
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            except Exception:
                continue

        return mask

    # COCO RLE format.
    if isinstance(seg, dict) and "counts" in seg:
        if not PYCOCOTOOLS_AVAILABLE:
            return mask

        try:
            rle_mask = coco_mask.decode(seg)
            if rle_mask.ndim == 3:
                rle_mask = np.max(rle_mask, axis=2)
            mask[rle_mask > 0] = 1
        except Exception:
            pass

        return mask

    return mask


def decode_instance_mask(json_path: str, image: Optional[np.ndarray]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    data = load_json_any(json_path)
    record = get_record_from_json(data)

    h, w = get_image_hw(record, image)
    inst_mask = np.zeros((h, w), dtype=np.int32)

    organ_name = extract_organ_from_record(record)

    annotations = record.get("annotations", None)
    if annotations is None and isinstance(data, list):
        annotations = data

    if annotations is None:
        annotations = []

    inst_id = 1

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        seg = ann.get("segmentation", None)
        binary = decode_segmentation_to_mask(seg, h, w, inst_id)

        if binary.sum() > 0:
            inst_mask[binary > 0] = inst_id
            inst_id += 1

    return inst_mask, organ_name, record


# ==============================================================================
# 4. Per-image statistics
# ==============================================================================
def compute_nearest_neighbor_stats(centroids: np.ndarray) -> Tuple[float, float, float]:
    if centroids is None or len(centroids) <= 1:
        return 0.0, 0.0, 0.0

    pts = centroids.astype(np.float32)

    try:
        if SKLEARN_AVAILABLE:
            tree = KDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn = dists[:, 1]
        elif SCIPY_AVAILABLE:
            tree = ScipyKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn = dists[:, 1]
        else:
            if len(pts) > 1000:
                return 0.0, 0.0, 0.0

            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist, np.inf)
            nn = np.min(dist, axis=1)

        nn_mean = safe_float(np.mean(nn), 0.0)
        nn_std = safe_float(np.std(nn), 0.0)
        nn_cv = safe_float(nn_std / (nn_mean + 1e-6), 0.0)
        return nn_mean, nn_std, nn_cv

    except Exception:
        return 0.0, 0.0, 0.0


def compute_image_stats(image_rgb: Optional[np.ndarray], inst_mask: np.ndarray) -> Dict[str, float]:
    if inst_mask is None or inst_mask.max() <= 0:
        h, w = inst_mask.shape[:2] if inst_mask is not None else (256, 256)
        return {
            "height": float(h),
            "width": float(w),
            "image_area": float(h * w),
            "nuclei_count": 0.0,
            "density_per_mpix": 0.0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "std_area": 0.0,
            "mean_eccentricity": 0.0,
            "mean_aspect_ratio": 1.0,
            "mean_solidity": 0.0,
            "nn_distance_mean": 0.0,
            "nn_distance_std": 0.0,
            "nn_distance_cv": 0.0,
            "mean_intensity": 0.0,
        }

    h, w = inst_mask.shape[:2]
    props = measure.regionprops(inst_mask)

    valid_props = [p for p in props if 10 < p.area < 100000]
    count = len(valid_props)

    if count == 0:
        return {
            "height": float(h),
            "width": float(w),
            "image_area": float(h * w),
            "nuclei_count": 0.0,
            "density_per_mpix": 0.0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "std_area": 0.0,
            "mean_eccentricity": 0.0,
            "mean_aspect_ratio": 1.0,
            "mean_solidity": 0.0,
            "nn_distance_mean": 0.0,
            "nn_distance_std": 0.0,
            "nn_distance_cv": 0.0,
            "mean_intensity": 0.0,
        }

    areas = np.array([p.area for p in valid_props], dtype=np.float32)
    eccentricities = np.array([p.eccentricity for p in valid_props], dtype=np.float32)
    solidities = np.array([p.solidity for p in valid_props], dtype=np.float32)

    aspect_ratios = []
    centroids = []

    for p in valid_props:
        y_min, x_min, y_max, x_max = p.bbox
        bw = max(1.0, float(x_max - x_min))
        bh = max(1.0, float(y_max - y_min))
        ar = max(bw / bh, bh / bw)
        aspect_ratios.append(ar)
        centroids.append([p.centroid[1], p.centroid[0]])

    aspect_ratios = np.array(aspect_ratios, dtype=np.float32)
    centroids = np.array(centroids, dtype=np.float32)

    nn_mean, nn_std, nn_cv = compute_nearest_neighbor_stats(centroids)

    density_per_mpix = float(count) / max(float(h * w) / 1_000_000.0, 1e-6)

    mean_intensity = 0.0
    if image_rgb is not None and image_rgb.ndim == 3:
        binary = inst_mask > 0
        if binary.sum() > 0:
            nuclei_pixels = image_rgb[binary]
            gray = (
                0.299 * nuclei_pixels[:, 0].astype(np.float32)
                + 0.587 * nuclei_pixels[:, 1].astype(np.float32)
                + 0.114 * nuclei_pixels[:, 2].astype(np.float32)
            )
            mean_intensity = safe_float(np.mean(gray), 0.0)

    return {
        "height": float(h),
        "width": float(w),
        "image_area": float(h * w),
        "nuclei_count": float(count),
        "density_per_mpix": safe_float(density_per_mpix, 0.0),
        "mean_area": safe_float(np.mean(areas), 0.0),
        "median_area": safe_float(np.median(areas), 0.0),
        "std_area": safe_float(np.std(areas), 0.0),
        "mean_eccentricity": safe_float(np.mean(eccentricities), 0.0),
        "mean_aspect_ratio": safe_float(np.mean(aspect_ratios), 1.0),
        "mean_solidity": safe_float(np.mean(solidities), 0.0),
        "nn_distance_mean": safe_float(nn_mean, 0.0),
        "nn_distance_std": safe_float(nn_std, 0.0),
        "nn_distance_cv": safe_float(nn_cv, 0.0),
        "mean_intensity": safe_float(mean_intensity, 0.0),
    }


# ==============================================================================
# 5. Thresholds and labeling
# ==============================================================================
def quantile(values: List[float], q: float, default: float = 0.0) -> float:
    arr = np.array([safe_float(v, np.nan) for v in values], dtype=np.float32)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def compute_train_thresholds(records: List[Dict[str, Any]]) -> Dict[str, float]:
    train_valid = [
        r for r in records
        if r["split"] == "train" and r["stats"]["nuclei_count"] > 0
    ]

    if len(train_valid) == 0:
        print("⚠️ No valid train records found. Falling back to all valid records for threshold computation.")
        train_valid = [
            r for r in records
            if r["stats"]["nuclei_count"] > 0
        ]

    size_values = [r["stats"]["mean_area"] for r in train_valid]
    density_values = [r["stats"]["density_per_mpix"] for r in train_valid]
    shape_values = [r["stats"]["mean_eccentricity"] for r in train_valid]
    arrangement_values = [r["stats"]["nn_distance_cv"] for r in train_valid]
    color_values = [r["stats"]["mean_intensity"] for r in train_valid if r["stats"]["mean_intensity"] > 0]

    thresholds = {
        "size_q33": quantile(size_values, 0.33, 250.0),
        "size_q66": quantile(size_values, 0.66, 600.0),

        "density_q33": quantile(density_values, 0.33, 50.0),
        "density_q66": quantile(density_values, 0.66, 200.0),

        "shape_q33": quantile(shape_values, 0.33, 0.60),
        "shape_q66": quantile(shape_values, 0.66, 0.85),

        "arrangement_q66": quantile(arrangement_values, 0.66, 0.60),

        "color_q50": quantile(color_values, 0.50, 160.0),
    }

    return thresholds


def label_3way(value: float, low_th: float, high_th: float) -> int:
    value = safe_float(value, 0.0)

    if low_th >= high_th:
        if value < low_th:
            return 0
        if value > high_th:
            return 2
        return 1

    if value <= low_th:
        return 0
    if value >= high_th:
        return 2
    return 1


def label_from_stats(stats: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[List[int], Dict[str, str]]:
    # Label order: [color, shape, arrangement, size, density]

    # Color: 0 deep, 1 light.
    # Higher grayscale intensity means lighter nuclei.
    color_label = 1 if stats["mean_intensity"] >= thresholds["color_q50"] else 0

    # Shape: low eccentricity -> round, middle -> oval, high -> elongated.
    shape_label = label_3way(
        stats["mean_eccentricity"],
        thresholds["shape_q33"],
        thresholds["shape_q66"],
    )

    # Arrangement: high nearest-neighbor CV -> clustered/disordered.
    arrangement_label = 1 if stats["nn_distance_cv"] >= thresholds["arrangement_q66"] else 0

    # Size.
    size_label = label_3way(
        stats["mean_area"],
        thresholds["size_q33"],
        thresholds["size_q66"],
    )

    # Density.
    density_label = label_3way(
        stats["density_per_mpix"],
        thresholds["density_q33"],
        thresholds["density_q66"],
    )

    labels = [
        int(color_label),
        int(shape_label),
        int(arrangement_label),
        int(size_label),
        int(density_label),
    ]

    visuals = {
        "color": LABEL_SPACES["color"][labels[0]],
        "shape": LABEL_SPACES["shape"][labels[1]],
        "arrangement": LABEL_SPACES["arrangement"][labels[2]],
        "size": LABEL_SPACES["size"][labels[3]],
        "density": LABEL_SPACES["density"][labels[4]],
    }

    return labels, visuals


# ==============================================================================
# 6. Prompt generation
# ==============================================================================
def build_prompts(organ_name: str, visuals: Dict[str, str]) -> Dict[str, str]:
    organ_name = normalize_organ_name(organ_name)
    med_info = DEFAULT_ORGAN_KNOWLEDGE.get(organ_name, DEFAULT_ORGAN_KNOWLEDGE["Generic"])

    context = med_info["context"]
    desc = med_info["desc"]

    color = visuals["color"]
    shape = visuals["shape"]
    arrangement = visuals["arrangement"]
    size = visuals["size"]
    density = visuals["density"]

    text_prompt = (
        f"Cell nuclei in {context}."
    )

    attribute_text = (
        f"H&E-stained {context} histopathology patch. "
        f"The nuclei are {color}, {size}, {density}, {arrangement}, "
        f"and {shape} in shape. "
        f"These attribute-aware prompts describe nuclear staining, size, density, "
        f"spatial arrangement, and morphology."
    )

    morphology_text = (
        f"H&E-stained {context} histopathology patch. "
        f"Focus on {shape} nuclear contours, {size} nuclei, {density}, "
        f"and {arrangement}. "
        f"Emphasize sharp nuclear boundaries, touching nuclei separation, "
        f"contour clarity, and instance-level delineation."
    )

    legacy_text_prompt = (
        f"Microscopic view of {density}, {size} {desc} "
        f"with {shape} features, in {context}."
    )

    return {
        "text_prompt": " ".join(text_prompt.split()),
        "attribute_text": " ".join(attribute_text.split()),
        "morphology_text": " ".join(morphology_text.split()),
        "legacy_text_prompt": " ".join(legacy_text_prompt.split()),
    }


# ==============================================================================
# 7. Generator
# ==============================================================================
class MedicalKnowledgeV2Generator:
    def __init__(self, data_root: str, output_path: str):
        self.data_root = data_root
        self.output_path = output_path
        self.records: List[Dict[str, Any]] = []
        self.kb: Dict[str, Any] = {}

    def scan_json_files(self) -> List[str]:
        search_path = os.path.join(self.data_root, "**", "*.json")
        all_files = glob.glob(search_path, recursive=True)

        filtered = []
        for path in all_files:
            name = os.path.basename(path).lower()
            if "knowledge" in name:
                continue
            if name.startswith("medical_knowledge"):
                continue
            filtered.append(path)

        filtered = sorted(filtered)
        return filtered

    def collect_records(self):
        json_files = self.scan_json_files()

        if len(json_files) == 0:
            raise RuntimeError(f"No annotation JSON files found under: {self.data_root}")

        print(f"🚀 Found {len(json_files)} annotation json files.")
        print("🔍 Phase 1: decoding masks and computing raw image-level statistics...")

        for json_path in tqdm(json_files, desc="Collect stats"):
            img_path = find_image_path(json_path)

            image_rgb = None
            if img_path is not None:
                image_bgr = cv2.imread(img_path)
                if image_bgr is not None:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            try:
                inst_mask, organ_raw, record_json = decode_instance_mask(json_path, image_rgb)
            except Exception as exc:
                print(f"⚠️ Failed to decode {json_path}: {exc}")
                continue

            rel_json = os.path.relpath(json_path, self.data_root).replace("\\", "/")
            rel_img = os.path.splitext(rel_json)[0] + ".png"

            if img_path is not None:
                rel_img_real = os.path.relpath(img_path, self.data_root).replace("\\", "/")
                rel_img = rel_img_real

            split = infer_split(rel_json, record_json)
            organ_name = normalize_organ_name(organ_raw)

            stats = compute_image_stats(image_rgb, inst_mask)

            self.records.append(
                {
                    "json_path": json_path,
                    "img_path": img_path,
                    "rel_img": rel_img,
                    "rel_json": rel_json,
                    "split": split,
                    "organ_id": organ_name,
                    "organ_idx": ORGAN_TO_ID.get(organ_name, ORGAN_TO_ID["Generic"]),
                    "stats": stats,
                }
            )

        if len(self.records) == 0:
            raise RuntimeError("No valid records were collected.")

    def build_organ_priors(self, enriched_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        organ_combo_counter = defaultdict(Counter)
        organ_visual_counter = {
            organ: {
                "color": Counter(),
                "shape": Counter(),
                "arrangement": Counter(),
                "size": Counter(),
                "density": Counter(),
            }
            for organ in DEFAULT_ORGAN_KNOWLEDGE.keys()
        }

        for rec in enriched_records:
            if rec["split"] != "train":
                continue

            organ = rec["organ_id"]
            labels = tuple(rec["attr_labels"])
            visuals = rec["visual_stats"]

            organ_combo_counter[organ][labels] += 1

            for key in ["color", "shape", "arrangement", "size", "density"]:
                organ_visual_counter[organ][key][visuals[key]] += 1

        priors = {}

        for organ in DEFAULT_ORGAN_KNOWLEDGE.keys():
            combo_counter = organ_combo_counter.get(organ, Counter())

            if len(combo_counter) > 0:
                label_tuple, count = combo_counter.most_common(1)[0]
                attr_labels = list(label_tuple)
            else:
                attr_labels = list(DEFAULT_ATTR_LABELS)

            visual_stats = {}
            for key in ["color", "shape", "arrangement", "size", "density"]:
                c = organ_visual_counter[organ][key]
                if len(c) > 0:
                    visual_stats[key] = c.most_common(1)[0][0]
                else:
                    visual_stats[key] = LABEL_SPACES[key][attr_labels[ATTRIBUTE_ORDER.index(key)]]

            priors[organ] = {
                "organ_idx": ORGAN_TO_ID.get(organ, ORGAN_TO_ID["Generic"]),
                "attr_labels": attr_labels,
                "visual_stats": visual_stats,
            }

        return priors

    def print_distribution_report(self, enriched_records: List[Dict[str, Any]]):
        train_records = [r for r in enriched_records if r["split"] == "train"]
        if len(train_records) == 0:
            train_records = enriched_records

        combo_counter = Counter()
        dim_counters = [Counter() for _ in range(5)]
        organ_counter = Counter()

        for rec in train_records:
            labels = tuple(rec["attr_labels"])
            combo_counter[labels] += 1
            organ_counter[rec["organ_id"]] += 1

            for i, v in enumerate(labels):
                dim_counters[i][v] += 1

        total = max(1, len(train_records))

        print("\n📊 [medical_knowledge_v2 attribute distribution: train split]")
        print(f"Train records: {len(train_records)}")

        print("\ncombo top20:")
        for combo, count in combo_counter.most_common(20):
            print(f"  {combo} {count} {count / total:.3f}")

        print("\nper-dim distribution:")
        for i, counter in enumerate(dim_counters):
            name = ATTRIBUTE_ORDER[i]
            print(f"  dim{i} {name}:")
            dim_total = max(1, sum(counter.values()))
            for label_id in sorted(counter.keys()):
                label_name = LABEL_SPACES[name][label_id]
                count = counter[label_id]
                print(f"    {label_id} ({label_name}): {count} {count / dim_total:.3f}")

        print("\norgan distribution top20:")
        for organ, count in organ_counter.most_common(20):
            print(f"  {organ}: {count} {count / total:.3f}")

    def run(self):
        self.collect_records()

        print("\n📐 Phase 2: computing train-only quantile thresholds...")
        thresholds = compute_train_thresholds(self.records)

        print("\n📊 [Train-only thresholds]")
        for k, v in thresholds.items():
            print(f"  {k}: {v:.6f}")

        print("\n🧬 Phase 3: generating v2 attribute labels and PromptNu-style prompts...")

        enriched_records = []

        for rec in tqdm(self.records, desc="Generate knowledge"):
            stats = rec["stats"]
            labels, visuals = label_from_stats(stats, thresholds)
            prompts = build_prompts(rec["organ_id"], visuals)

            visual_stats = {
                "color": visuals["color"],
                "shape": visuals["shape"],
                "arrangement": visuals["arrangement"],
                "size": visuals["size"],
                "density": visuals["density"],

                "nuclei_count": stats["nuclei_count"],
                "density_per_mpix": stats["density_per_mpix"],
                "mean_area": stats["mean_area"],
                "median_area": stats["median_area"],
                "std_area": stats["std_area"],
                "mean_eccentricity": stats["mean_eccentricity"],
                "mean_aspect_ratio": stats["mean_aspect_ratio"],
                "mean_solidity": stats["mean_solidity"],
                "nn_distance_mean": stats["nn_distance_mean"],
                "nn_distance_std": stats["nn_distance_std"],
                "nn_distance_cv": stats["nn_distance_cv"],
                "mean_intensity": stats["mean_intensity"],
            }

            entry = {
                "organ_id": rec["organ_id"],
                "organ_idx": rec["organ_idx"],
                "split": rec["split"],

                # Main supervision for PNuRL.
                # Label order: [color, shape, arrangement, size, density]
                "attr_labels": labels,
                "attribute_order": ATTRIBUTE_ORDER,

                # Human-readable and numeric stats.
                "visual_stats": visual_stats,

                # PromptNu-style prompts.
                "text_prompt": prompts["text_prompt"],
                "attribute_text": prompts["attribute_text"],
                "morphology_text": prompts["morphology_text"],

                # Kept for backward compatibility / debug.
                "legacy_text_prompt": prompts["legacy_text_prompt"],

                "source": "train_quantile_full_image_mask_stats",
            }

            self.kb[rec["rel_img"]] = entry

            enriched = dict(rec)
            enriched.update(entry)
            enriched_records.append(enriched)

        organ_priors = self.build_organ_priors(enriched_records)

        meta = {
            "version": "promptnu_freqpath_v2",
            "description": (
                "PromptNu-style medical knowledge generated from full-image instance masks. "
                "Thresholds are computed only from train split using quantiles. "
                "attr_labels order: [color, shape, arrangement, size, density]."
            ),
            "attribute_order": ATTRIBUTE_ORDER,
            "label_spaces": LABEL_SPACES,
            "num_classes_per_attr": [
                len(LABEL_SPACES["color"]),
                len(LABEL_SPACES["shape"]),
                len(LABEL_SPACES["arrangement"]),
                len(LABEL_SPACES["size"]),
                len(LABEL_SPACES["density"]),
            ],
            "train_thresholds": thresholds,
            "organ_to_id": ORGAN_TO_ID,
            "organ_priors": organ_priors,
        }

        self.kb["__meta__"] = meta

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.kb, f, indent=4, ensure_ascii=False)

        self.print_distribution_report(enriched_records)

        print(f"\n✅ Success. medical_knowledge_v2 saved to: {self.output_path}")
        print("\nNext checks:")
        print("  1. top1 combo should not dominate excessively.")
        print("  2. dim1/shape, dim3/size, dim4/density should have multi-class distribution.")
        print("  3. DataLoader should read attr_labels / attribute_text directly from this v2 file.")


# ==============================================================================
# 8. CLI
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PromptNu/FreqPath-style medical_knowledge_v2.json"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/PanNuke",
        help="Dataset root containing train/test annotation json and images.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path. Default: <data_root>/medical_knowledge_v2.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.data_root, "medical_knowledge_v2.json")

    generator = MedicalKnowledgeV2Generator(
        data_root=args.data_root,
        output_path=output_path,
    )
    generator.run()


if __name__ == "__main__":
    main()