import argparse
import os
import math
import cv2
import json
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from collections import defaultdict
import multiprocessing as mp

# === 核心防死锁机制 ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 👇 新增：通用环境变量清洗器（自动剔除所有非 ASCII 脏字符）
for _key, _value in list(os.environ.items()):
    if isinstance(_value, str):
        # 强制只保留标准 ASCII 字符 (0-127)，肉眼不可见的幽灵字符会被瞬间蒸发
        os.environ[_key] = "".join(c for c in _value if ord(c) < 128).strip()

import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.modeling.sam import TextSam
from metrics import SegMetrics

from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as skimage_label


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

ID_TO_ORGAN = {v: k for k, v in ORGAN_TO_ID.items()}


# ==================================================================================================
# 0. Prompt 工具：测试阶段只使用静态 organ prompt，不使用 GT mask-derived 动态属性
# ==================================================================================================
def format_organ_name(organ_name: str) -> str:
    if organ_name is None:
        return "generic"

    name = str(organ_name).strip()
    if not name:
        return "generic"

    special_map = {
        "Adrenal_gland": "adrenal gland",
        "Bile-duct": "bile duct",
        "HeadNeck": "head and neck",
        "Ovarian": "ovary",
        "Pancreatic": "pancreas",
        "Generic": "generic",
    }

    if name in special_map:
        return special_map[name]

    name = name.replace("_", " ").replace("-", " ")
    return name.lower()


def build_test_prompts(organ_name: str, prompt_mode: str = "organ_static"):
    """
    测试阶段 prompt。

    注意:
        不使用 mask 统计出来的 shape / density / arrangement。
        这样避免测试时 GT 信息泄漏。

    prompt_mode:
        base:
            全部使用 Cell nuclei。
        generic:
            使用通用病理图像上下文。
        organ_static:
            使用 organ-aware 静态 prompt，默认推荐。
    """
    prompt_mode = str(prompt_mode).lower().strip()
    organ_text = format_organ_name(organ_name)

    base_prompt = "Cell nuclei"

    if prompt_mode == "base":
        return base_prompt, base_prompt, base_prompt

    if prompt_mode == "generic" or organ_text == "generic":
        text_prompt = "Cell nuclei in H&E-stained histopathology tissue."
        attribute_text = (
            "H&E-stained histopathology patch. "
            "The image contains cell nuclei. "
            "Focus on nuclear regions without using crop-level mask-derived attributes."
        )
        morphology_text = (
            "H&E-stained histopathology patch. "
            "Focus on nuclear boundaries, touching nuclei, and instance-level delineation."
        )
        return text_prompt, attribute_text, morphology_text

    text_prompt = f"Cell nuclei in {organ_text} tissue."
    attribute_text = (
        f"H&E-stained {organ_text} histopathology patch. "
        f"The image contains cell nuclei in {organ_text} tissue. "
        f"This prompt provides organ context without using crop-level mask-derived attributes."
    )
    morphology_text = (
        f"H&E-stained {organ_text} histopathology patch. "
        f"Focus on nuclear boundaries, touching nuclei, and instance-level delineation."
    )

    return text_prompt, attribute_text, morphology_text


def extract_organ_from_json(json_path: str):
    organ_name = "Generic"
    organ_id = 20

    if not os.path.exists(json_path):
        return organ_name, organ_id

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict):
            if "organ_id" in data:
                organ_name = data.get("organ_id", "Generic")
                organ_id = ORGAN_TO_ID.get(organ_name, 20)
            elif "organ_idx" in data:
                organ_id = int(data.get("organ_idx", 20))
                organ_name = ID_TO_ORGAN.get(organ_id, "Generic")
    except Exception:
        organ_name = "Generic"
        organ_id = 20

    return organ_name, organ_id


# ==================================================================================================
# 1. Checkpoint / position embedding tools
# ==================================================================================================
def resize_pos_embed(state_dict, model_state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                if "pos_embed" in k:
                    v = v.permute(0, 3, 1, 2)
                    v = F.interpolate(
                        v,
                        size=model_state_dict[k].shape[1:3],
                        mode="bicubic",
                        align_corners=False,
                    )
                    v = v.permute(0, 2, 3, 1)
                elif "rel_pos" in k:
                    v = v.unsqueeze(0).permute(0, 2, 1)
                    target_len = model_state_dict[k].shape[0]
                    v = F.interpolate(
                        v,
                        size=target_len,
                        mode="linear",
                        align_corners=False,
                    )
                    v = v.permute(0, 2, 1).squeeze(0)

            new_state_dict[k] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict

    return {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }


def load_model_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    state_dict = strip_module_prefix(state_dict)
    state_dict = resize_pos_embed(state_dict, model.state_dict())

    load_ret = model.load_state_dict(state_dict, strict=False)

    missing_keys = getattr(load_ret, "missing_keys", [])
    unexpected_keys = getattr(load_ret, "unexpected_keys", [])

    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")

    return model


# ==================================================================================================
# 2. 核心后处理
# ==================================================================================================
def hover_post_process(prob_map, hv_map, prob_thresh=0.40, marker_thresh=0.45, min_marker_size=12):
    mask = prob_map > prob_thresh
    mask = binary_fill_holes(mask)

    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32)

    v_map = hv_map[0].astype(np.float32)
    h_map = hv_map[1].astype(np.float32)

    diff_v = np.gradient(v_map, axis=0)
    diff_h = np.gradient(h_map, axis=1)
    sobel_mag = np.sqrt(diff_v ** 2 + diff_h ** 2)

    marker_map = prob_map - sobel_mag
    marker_map = (marker_map > marker_thresh) & mask
    marker_map = remove_small_objects(marker_map, min_size=int(min_marker_size))

    markers = skimage_label(marker_map).astype(np.int32)

    if markers.max() == 0:
        markers = skimage_label(mask).astype(np.int32)

    inst_map = watershed(-prob_map, markers, mask=mask)

    inst_map = remove_small_objects(inst_map, min_size=15)
    inst_map = inst_map.astype(np.int32)

    return inst_map


# ==================================================================================================
# 3. 8-fold TTA batch inference
# ==================================================================================================
def tta_inference_8x_batch(model, image_rgb, organ_id, organ_name, args):
    device = args.device
    input_size = (args.image_size, args.image_size)

    transforms = [
        (None, 0),
        (1, 0),
        (0, 0),
        (-1, 0),
        (None, 1),
        (1, 1),
        (0, 1),
        (-1, 1),
    ]

    text_prompt, attribute_text, morphology_text = build_test_prompts(
        organ_name=organ_name,
        prompt_mode=args.prompt_mode,
    )

    img_list = []
    for f_code, r_k in transforms:
        img_t = image_rgb.copy()

        if f_code is not None:
            img_t = cv2.flip(img_t, f_code)

        if r_k > 0:
            img_t = np.rot90(img_t, k=r_k)

        img_t = cv2.resize(img_t, input_size)
        img_list.append(torch.from_numpy(img_t).permute(2, 0, 1).float())

    batch_img = torch.stack(img_list).to(device)

    all_probs = []
    all_hvs = []
    first_attr_logits = {}

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_samples = []

        for i in range(len(transforms)):
            input_samples.append(
                {
                    "image": batch_img[i],
                    "original_size": input_size,
                    "organ_id": int(organ_id),
                    "text_prompt": text_prompt,
                    "attribute_text": attribute_text,
                    "morphology_text": morphology_text,
                    "attr_labels": None,
                }
            )

        outputs = model(input_samples, multimask_output=True)

        for i in range(len(transforms)):
            out = outputs[i]

            iou_predictions = out["iou_predictions"]
            if iou_predictions.ndim == 2:
                iou_predictions = iou_predictions.squeeze(0)

            best_idx = torch.argmax(iou_predictions).item()

            masks = out["masks"]
            if masks.dim() == 4:
                prob = torch.sigmoid(masks[0, best_idx])
            elif masks.dim() == 3:
                prob = torch.sigmoid(masks[best_idx])
            else:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")

            hv_raw = out.get("hv_logits", None)

            if hv_raw is not None:
                if hv_raw.dim() == 3:
                    hv_raw = hv_raw.unsqueeze(0)

                hv_raw = torch.tanh(hv_raw.float())
                hv = F.interpolate(
                    hv_raw,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                hv = torch.zeros((2, input_size[0], input_size[1]), device=device)

            if i == 0:
                first_attr_logits = out.get("attr_logits", {}) or {}

            f_code, r_k = transforms[i]

            # Inverse rotation first because forward order is flip -> rotation.
            if r_k == 1:
                prob = torch.rot90(prob, k=-1, dims=[0, 1])
                hv = torch.rot90(hv, k=-1, dims=[1, 2])

                # HV vector inverse rotation.
                v_new = hv[1].clone()
                h_new = -hv[0].clone()
                hv[0], hv[1] = v_new, h_new

            # Inverse flip.
            if f_code == 1:
                prob = torch.flip(prob, [1])
                hv = torch.flip(hv, [2])
                hv[1] = -hv[1]
            elif f_code == 0:
                prob = torch.flip(prob, [0])
                hv = torch.flip(hv, [1])
                hv[0] = -hv[0]
            elif f_code == -1:
                prob = torch.flip(prob, [0, 1])
                hv = torch.flip(hv, [1, 2])
                hv = -hv

            all_probs.append(prob)
            all_hvs.append(hv)

    avg_prob = torch.stack(all_probs).mean(0).cpu().float().numpy()
    avg_hv = torch.stack(all_hvs).mean(0).cpu().float().numpy()

    return avg_prob, avg_hv, first_attr_logits


# ==================================================================================================
# 4. Sliding window inference
# ==================================================================================================
def get_gaussian_kernel(size, sigma=1.0):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2)))
    return kernel.astype(np.float32)


def sliding_window_inference(model, image_rgb, organ_id, organ_name, args, patch_size=256, overlap=0.8):
    h, w = image_rgb.shape[:2]

    stride = max(1, int(patch_size * (1 - overlap)))

    pad_h = 0 if h % stride == 0 else stride - (h % stride)
    pad_w = 0 if w % stride == 0 else stride - (w % stride)

    pad_h = max(pad_h, patch_size - h) if h < patch_size else pad_h
    pad_w = max(pad_w, patch_size - w) if w < patch_size else pad_w

    padded_img = np.pad(
        image_rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )

    pad_h_full, pad_w_full = padded_img.shape[:2]

    canvas_prob = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)
    canvas_hv = np.zeros((2, pad_h_full, pad_w_full), dtype=np.float32)
    canvas_weight = np.zeros((pad_h_full, pad_w_full), dtype=np.float32)

    weight_mask = get_gaussian_kernel(patch_size, sigma=0.33)
    accumulated_size_logits = None

    for y in range(0, pad_h_full - patch_size + 1, stride):
        for x in range(0, pad_w_full - patch_size + 1, stride):
            patch = padded_img[y:y + patch_size, x:x + patch_size, :]

            prob_512, hv_512, attr_logits = tta_inference_8x_batch(
                model=model,
                image_rgb=patch,
                organ_id=organ_id,
                organ_name=organ_name,
                args=args,
            )

            prob_patch = cv2.resize(
                prob_512,
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )

            hv_v_patch = cv2.resize(
                hv_512[0],
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )

            hv_h_patch = cv2.resize(
                hv_512[1],
                (patch_size, patch_size),
                interpolation=cv2.INTER_LINEAR,
            )

            canvas_prob[y:y + patch_size, x:x + patch_size] += prob_patch * weight_mask
            canvas_hv[0, y:y + patch_size, x:x + patch_size] += hv_v_patch * weight_mask
            canvas_hv[1, y:y + patch_size, x:x + patch_size] += hv_h_patch * weight_mask
            canvas_weight[y:y + patch_size, x:x + patch_size] += weight_mask

            if isinstance(attr_logits, dict) and "size" in attr_logits:
                size_logits = attr_logits["size"].detach().cpu()

                if accumulated_size_logits is None:
                    accumulated_size_logits = size_logits.clone()
                else:
                    accumulated_size_logits += size_logits

    canvas_prob /= (canvas_weight + 1e-8)
    canvas_hv /= (canvas_weight + 1e-8)

    final_prob = canvas_prob[:h, :w]
    final_hv = canvas_hv[:, :h, :w]

    dynamic_min_size = args.min_marker_size

    if accumulated_size_logits is not None:
        if accumulated_size_logits.ndim > 1:
            mean_logits = accumulated_size_logits.mean(dim=0)
        else:
            mean_logits = accumulated_size_logits

        pred_size_idx = torch.argmax(mean_logits).item()
        dynamic_min_size = {0: 12, 1: 25, 2: 38}.get(pred_size_idx, args.min_marker_size)

    return final_prob, final_hv, dynamic_min_size


# ==================================================================================================
# 5. GT loading
# ==================================================================================================
def load_filtered_gt(img_path):
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        annotations = data.get("annotations", []) if isinstance(data, dict) else data

        if not annotations:
            return None

        h, w = None, None

        if isinstance(data, dict):
            h, w = data.get("height"), data.get("width")

        if h is None or w is None:
            first_seg = annotations[0].get("segmentation", {})
            if isinstance(first_seg, dict) and "size" in first_seg:
                h, w = first_seg["size"]
            else:
                h, w = 1000, 1000

        instance_map = np.zeros((int(h), int(w)), dtype=np.int32)

        for idx, ann in enumerate(annotations):
            seg = ann.get("segmentation")
            if not seg:
                continue

            if isinstance(seg, list):
                for poly in seg:
                    poly_np = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    if poly_np.shape[0] >= 3:
                        poly_np = np.round(poly_np).astype(np.int32)
                        cv2.fillPoly(instance_map, [poly_np], idx + 1)
            elif isinstance(seg, dict) and "counts" in seg:
                binary_mask = mask_utils.decode(seg)
                instance_map[binary_mask > 0] = idx + 1

        return instance_map

    except Exception:
        return None


# ==================================================================================================
# 6. Prediction saving
# ==================================================================================================
def save_prediction(pred_mask, img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(img_path))[0]

    npy_path = os.path.join(output_dir, f"{base}_inst.npy")
    png_path = os.path.join(output_dir, f"{base}_inst.png")

    np.save(npy_path, pred_mask.astype(np.int32))

    max_val = int(pred_mask.max())
    if max_val <= 65535:
        cv2.imwrite(png_path, pred_mask.astype(np.uint16))
    else:
        # PNG uint16 不够时，仍保留 npy 作为完整实例图。
        vis = (pred_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(png_path, vis)


# ==================================================================================================
# 7. Parallel worker
# ==================================================================================================
def process_chunk(worker_id, image_files_chunk, args):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("CUDA is required for this testing pipeline, but no GPU was detected.")

    gpu_id = worker_id % num_gpus
    device = torch.device(f"cuda:{gpu_id}")
    args.device = device

    fine_tuned_ckpt = args.checkpoint

    # 构建 vanilla SAM。这里不加载 checkpoint，后面直接加载 TextSam 权重。
    args.checkpoint = None
    vanilla_sam = sam_model_registry[args.model_type](args)
    args.checkpoint = fine_tuned_ckpt

    model = TextSam(
        image_encoder=vanilla_sam.image_encoder,
        prompt_encoder=vanilla_sam.prompt_encoder,
        mask_decoder=vanilla_sam.mask_decoder,
        clip_model_name=args.clip_model,
        num_organs=args.num_organs,
        num_heads=args.num_heads,

        # OT 已从当前主线移除，测试固定关闭。
        sg_epsilon=0.05,
        sg_iters=3,
        use_pnurl=args.use_pnurl,
        use_coop=args.use_coop,
        use_ot=False,
        use_asr=args.use_asr,
    ).to(device)

    del vanilla_sam

    model = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    chunk_metrics = defaultdict(list)

    pbar = tqdm(
        image_files_chunk,
        desc=f"Worker {worker_id} (GPU {gpu_id})",
        position=worker_id,
        leave=False,
    )

    for img_path in pbar:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        json_path = os.path.splitext(img_path)[0] + ".json"
        organ_name, organ_id = extract_organ_from_json(json_path)

        prob, hv, dynamic_min_size = sliding_window_inference(
            model=model,
            image_rgb=image_rgb,
            organ_id=organ_id,
            organ_name=organ_name,
            args=args,
            patch_size=args.patch_size,
            overlap=args.overlap,
        )

        pred_mask = hover_post_process(
            prob,
            hv,
            prob_thresh=args.prob_thresh,
            marker_thresh=args.marker_thresh,
            min_marker_size=dynamic_min_size,
        )

        if pred_mask.max() == 0:
            fallback_mask = prob > args.prob_thresh
            fallback_mask = binary_fill_holes(fallback_mask)
            pred_mask = skimage_label(fallback_mask).astype(np.int32)
            pred_mask = remove_small_objects(pred_mask, min_size=15).astype(np.int32)

        if args.save_pred:
            save_prediction(pred_mask, img_path, args.output_dir)

        gt_mask = load_filtered_gt(img_path)

        if gt_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint16),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)

            res = SegMetrics(pred_mask, gt_mask, args.metrics)
            for k, v in res.items():
                chunk_metrics[k].append(v)

    return dict(chunk_metrics)


# ==================================================================================================
# 8. Args
# ==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FreqPath-SAM Inference & Testing")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--output_dir", type=str, default="test_predictions")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.8)

    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--num_organs", type=int, default=21)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder_adapter", action="store_true", default=True)

    parser.add_argument("--use_pnurl", action="store_true", default=False)
    parser.add_argument("--use_coop", action="store_true", default=False)
    parser.add_argument("--use_asr", action="store_true", default=False)

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="organ_static",
        choices=["base", "generic", "organ_static"],
        help="Testing prompt mode. Do not use dynamic mask-derived prompt in test.",
    )

    parser.add_argument("--prob_thresh", type=float, default=0.40)
    parser.add_argument("--marker_thresh", type=float, default=0.45)
    parser.add_argument("--min_marker_size", type=int, default=12)

    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "mAJI", "mPQ"])

    return parser.parse_args()


# ==================================================================================================
# 9. Main
# ==================================================================================================
def main(args):
    mp.set_start_method("spawn", force=True)

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"data_path does not exist or is not a directory: {args.data_path}")

    image_files = [
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if f.lower().endswith((".png", ".tif", ".tiff"))
    ]

    image_files = sorted(image_files)

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in {args.data_path}")

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("CUDA is required for this testing pipeline, but no GPU was detected.")

    workers_per_gpu = max(1, int(args.workers_per_gpu))
    num_workers = min(num_gpus * workers_per_gpu, len(image_files))

    chunk_size = math.ceil(len(image_files) / num_workers)
    chunks = [
        image_files[i:i + chunk_size]
        for i in range(0, len(image_files), chunk_size)
    ]

    print(f"\n🚀 System Detected {num_gpus} GPUs. Launching {len(chunks)} parallel Workers.")
    print(f"🔥 Testing Pipeline: overlap={args.overlap}, patch_size={args.patch_size}, TTA=8x, MultiMask=ON")
    print(f"🧠 Prompt mode: {args.prompt_mode}")
    print(f"🧩 Modules: use_asr={args.use_asr}, use_pnurl={args.use_pnurl}, use_coop={args.use_coop}, use_ot=False")

    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append((i, chunk, args))

    all_metrics = defaultdict(list)

    with mp.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(process_chunk, tasks)

    for res in results:
        for k, v in res.items():
            all_metrics[k].extend(v)

    print("\n" + "🌟" * 15)
    print("📊 Final Results:")

    for k in args.metrics:
        values = all_metrics.get(k, [])
        if len(values) == 0:
            print(f"{k:>10}: N/A")
        else:
            print(f"{k:>10}: {np.mean(values):.4f}")

    print("🌟" * 15 + "\n")


if __name__ == "__main__":
    main(parse_args())