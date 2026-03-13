import os
import sys
import argparse
import uuid
from datetime import datetime
import torch
import torch.multiprocessing as mp
from PIL import Image

from omegaconf import OmegaConf

# =========================
# Project path
# =========================
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for r in project_roots:
    if r not in sys.path:
        sys.path.insert(0, r)

# =========================
# Original dependencies (preserved fully)
# =========================
from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (
    AutoencoderKLWan3_8,
    AutoencoderKLWan,
    WanT5EncoderModel,
    AutoTokenizer,
    Wan2_2Transformer3DModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2TI2VPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
    save_videos_grid_last_frame,
    get_image_to_video_latent_with_pad,
    save_videos_grid_with_crop
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler

# =========================
# Parameters
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input_root", default='/pfs/zhengshenghe/cof/data/test/restoration/LQ')
    p.add_argument("--output_root", default="/pfs/zhengshenghe/cof/VideoX-Fun/Restoration/inference")

    p.add_argument("--model_name", default="/pfs/shared_models/Wan2.2-TI2V-5B")
    p.add_argument("--config_path", default="/pfs/zhengshenghe/f-train/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml")

    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--video_length", type=int, default=9, help="Length of video in frames")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--run_name", default="naive", help="Second level output folder")

    return p.parse_args()

# =========================
# Output directory (modified version)
# =========================
def build_output_dir(root, model, run_name="naive"):
    model_short = os.path.basename(model.rstrip("/"))
    out = os.path.join(root, run_name, model_short)
    os.makedirs(out, exist_ok=True)
    return out

# =========================
# Dynamic sample_size calculation
# =========================
def calculate_dynamic_sample_size(image_path):
    """
    Compute the sample size of the image rounded up to the nearest multiple of 32, and return the original size.

    Returns:
        tuple: (sample_size [height, width], original_size (width, height))
    """
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Compute the nearest multiple of 32
    padded_height = ((original_height + 31) // 32) * 32
    padded_width = ((original_width + 31) // 32) * 32

    sample_size = [padded_height, padded_width]
    original_size = (original_width, original_height)
    return sample_size, original_size

def calculate_dynamic_sample_size_with_limit(image_path):
    """
    Compute the sample size of an image, supporting scaling for ultra-large images.
    
    - If the image height > 2560 or width > 1440, scale it to the nearest size close to 2560×1440 while preserving the original aspect ratio, ensuring both height and width are divisible by 32; in this case, original_size is returned as None.

    - Otherwise, use the original logic: round up to the nearest multiple of 32 and return the original size.

    Returns:
        tuple: (sample_size [height, width], original_size (width, height) or None)
    """
    img = Image.open(image_path)
    original_width, original_height = img.size

    if original_height > 1440 or original_width > 2560:
        target_area = min(2560 * 1440, original_width * original_height)
        aspect_ratio = original_width / original_height

        new_height = (target_area / aspect_ratio) ** 0.5
        new_width = new_height * aspect_ratio

        padded_height = int(new_height // 32) * 32
        padded_width = int(new_width // 32) * 32

        padded_height = max(32, padded_height)
        padded_width = max(32, padded_width)

        sample_size = [padded_height, padded_width]
        original_size = None
    else:
        padded_height = ((original_height + 31) // 32) * 32
        padded_width = ((original_width + 31) // 32) * 32

        sample_size = [padded_height, padded_width]
        original_size = (original_width, original_height)

    return sample_size, original_size

def collect_tasks(root):
    """
    Collect all image tasks and compute the pixel values of each image for load balancing.
    Returns:
        list: [(folder_name, img_path, pixel_load), ...]
    """
    tasks = []
    exts = (".png", ".jpg", ".jpeg")

    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue

        for f in sorted(os.listdir(p)):
            if f.lower().endswith(exts):
                img_path = os.path.join(p, f)
                img = Image.open(img_path)
                original_width, original_height = img.size
                if original_height > 1440 or original_width > 2560:
                    pixel_load = 2560 * 1440
                else:
                    pixel_load = original_width * original_height

                tasks.append((d, img_path, pixel_load))

    return tasks

def filter_completed_tasks(tasks, output_root):
    """
    Quickly filter out completed tasks.

    Args:
        tasks: [(folder_name, img_path, pixel_load), ...]
        output_root 

    Returns:
        list: List of unfinished tasks.
    """
    pending_tasks = []
    for folder_name, img_path, pixel_load in tasks:
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_root, folder_name, base + ".mp4")

        # 只保留未完成的任务
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            pending_tasks.append((folder_name, img_path, pixel_load))

    return pending_tasks

def balance_tasks(tasks, num_gpus):
    """
    Use a greedy algorithm to evenly distribute tasks across GPUs.

    Args:
        tasks: [(folder_name, img_path, pixel_load), ...]
        num_gpus

    Returns:
        list: [gpu0_tasks, gpu1_tasks, ...], 每个gpu_tasks是 [(folder_name, img_path), ...]
    """
    sorted_tasks = sorted(tasks, key=lambda x: x[2], reverse=True)

    gpu_tasks = [[] for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus

    
    for folder_name, img_path, pixel_load in sorted_tasks:
        
        min_load_gpu = gpu_loads.index(min(gpu_loads))
        
        gpu_tasks[min_load_gpu].append((folder_name, img_path))
        
        gpu_loads[min_load_gpu] += pixel_load

    return gpu_tasks

# =========================
# Worker
# =========================
def worker(rank, args, balanced_tasks, output_root):
    gpu_tasks = balanced_tasks[rank]

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    GPU_memory_mode ="model_full_load"

    enable_teacache = False
    teacache_threshold = 0.15
    num_skip_start_steps = 5
    teacache_offload = False

    cfg_skip_ratio = 0.15

    sampler_name = "Flow_Unipc"
    shift = 5

    video_length = args.video_length
    fps = 24

    weight_dtype = torch.bfloat16

    prompt = (
        "A restoration-focused video strictly based on the input image. "
        "The camera is completely static with no movement, no zoom, and no rotation. "
        "The original composition, objects, layout, and perspective are preserved exactly. "
        "Focus on visual restoration and enhancement: remove noise, reduce blur, eliminate rain artifacts, "
        "remove compression artifacts, and improve clarity, sharpness, and fine details while maintaining "
        "natural textures, accurate colors, and balanced lighting. "
        "Only extremely subtle and natural temporal consistency is allowed. "
        "The video should appear stable, clean, and realistic, as if the input image has been gently restored over time."
    )
    negative_prompt = (
        "camera movement, panning, tilting, zooming, rotation, "
        "scene change, object movement, new objects, object deformation, "
        "style change, artistic style, illustration, painting, cartoon, "
        "over-saturated colors, overexposure, underexposure, "
        "motion blur, jitter, flickering, shaking, "
        "low quality, worst quality, noise, blur, rain, fog, "
        "compression artifacts, jpeg artifacts, aliasing, "
        "text, subtitles, watermark, logo, "
        "distorted anatomy, extra limbs, duplicated objects, "
        "exaggerated motion, creative animation"
    )
    guidance_scale = 6.0
    num_inference_steps = 50

    # ========= 初始化 =========
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # device = set_multi_gpus_devices(1, 1)
    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    # ========= Transformer =========
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"
            ),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # ========= VAE =========
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]

    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(
            args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    # ========= Tokenizer / Text Encoder =========
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model_name,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["text_encoder_kwargs"].get(
                "text_encoder_subpath", "text_encoder"
            ),
        ),
        additional_kwargs=OmegaConf.to_container(
            config["text_encoder_kwargs"]
        ),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # ========= Scheduler =========
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]

    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config["scheduler_kwargs"]["shift"] = 1

    scheduler = Chosen_Scheduler(
        **filter_kwargs(
            Chosen_Scheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
    )

    # ========= Pipeline =========
    pipeline = Wan2_2TI2VPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # ========= GPU memory mode =========
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:  # model_full_load
        pipeline.to(device=device)

    # ========= TeaCache / CFG Skip =========
    if enable_teacache:
        coefficients = get_teacache_coefficients(args.model_name)
        pipeline.transformer.enable_teacache(
            coefficients,
            num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
            offload=teacache_offload,
        )

    pipeline.transformer.enable_cfg_skip(
        cfg_skip_ratio, num_inference_steps
    )

    generator = torch.Generator(device=device).manual_seed(
        args.seed + rank
    )

    for folder_name, png in gpu_tasks:
        folder_out = os.path.join(output_root, folder_name)
        os.makedirs(folder_out, exist_ok=True)

        base = os.path.splitext(os.path.basename(png))[0]
        out_path = os.path.join(folder_out, base + ".mp4")

        sample_size, original_size = calculate_dynamic_sample_size_with_limit(png)
        print(f"[GPU {rank}] Processing {png}: original_size={original_size}, sample_size={sample_size}")
        video, mask, _ = get_image_to_video_latent_with_pad(
            png,
            None,
            video_length=video_length,
            sample_size=sample_size,
        )

        sample = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_frames=video_length,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            video=video,
            mask_video=mask,
            shift=shift,
        ).videos

        torch.cuda.synchronize()
        save_videos_grid_with_crop(sample, out_path, original_size=original_size,fps=fps)
        del sample
        torch.cuda.empty_cache()


# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()
    output_root = build_output_dir(args.output_root, args.model_name, run_name=args.run_name)

    
    all_tasks = collect_tasks(args.input_root)
    print(f"Total tasks collected: {len(all_tasks)}")

    
    pending_tasks = filter_completed_tasks(all_tasks, output_root)
    print(f"Pending tasks after filtering: {len(pending_tasks)}")

    
    balanced_tasks = balance_tasks(pending_tasks, args.num_gpus)

    
    for i, gpu_task_list in enumerate(balanced_tasks):
        print(f"GPU {i}: {len(gpu_task_list)} tasks")

    
    mp.spawn(
        worker,
        args=(args, balanced_tasks, output_root),
        nprocs=args.num_gpus,
    )
