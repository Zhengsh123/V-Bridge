import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm import tqdm

def matlab_imresize(img, target_size):
    target_h, target_w = target_size
    h, w = img.shape[:2]

    zoom_h = target_h / h
    zoom_w = target_w / w

    if img.ndim == 3:
        resized = np.stack([
            zoom(img[:, :, c], (zoom_h, zoom_w), order=3)
            for c in range(3)
        ], axis=2)
    else:
        resized = zoom(img, (zoom_h, zoom_w), order=3)

    return np.clip(resized, 0, 255).astype(np.uint8)


def matlab_rgb2y(img):
    img = img.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    Y = 16 + (65.481 * R + 128.553 * G + 24.966 * B) / 255
    return Y


def matlab_style_ssim(img1, img2):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    K1 = 0.01
    K2 = 0.03
    L = 255

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = cv2.getGaussianKernel(11, 1.5)
    window = window @ window.T
    window /= np.sum(window)

    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_CONSTANT)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_CONSTANT)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, -1, window, borderType=cv2.BORDER_CONSTANT) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window, borderType=cv2.BORDER_CONSTANT) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_CONSTANT) - mu1_mu2

    pad = 5
    mu1_sq = mu1_sq[pad:-pad, pad:-pad]
    mu2_sq = mu2_sq[pad:-pad, pad:-pad]
    mu1_mu2 = mu1_mu2[pad:-pad, pad:-pad]
    sigma1_sq = sigma1_sq[pad:-pad, pad:-pad]
    sigma2_sq = sigma2_sq[pad:-pad, pad:-pad]
    sigma12 = sigma12[pad:-pad, pad:-pad]

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return np.mean(ssim_map)


def matlab_style_psnr(img1, img2):

    diff = img1.astype(np.float64) - img2.astype(np.float64)
    rmse = np.sqrt(np.mean(diff ** 2))

    if rmse == 0:
        return float('inf')

    return 20 * np.log10(255.0 / rmse)

def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read last frame: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calc_metrics(video_frame, gt_frame):

    h, w = gt_frame.shape[:2]
    video_frame = matlab_imresize(video_frame, (h, w))

    # Y 通道
    y_video = matlab_rgb2y(video_frame)
    y_gt = matlab_rgb2y(gt_frame)

    ssim_y = matlab_style_ssim(y_video, y_gt)
    psnr_y = matlab_style_psnr(y_video, y_gt)

    # RGB 三通道平均
    ssim_rgb = 0
    psnr_rgb = 0
    for c in range(3):
        ssim_rgb += matlab_style_ssim(video_frame[:, :, c], gt_frame[:, :, c])
        psnr_rgb += matlab_style_psnr(video_frame[:, :, c], gt_frame[:, :, c])

    ssim_rgb /= 3.0
    psnr_rgb /= 3.0

    return ssim_y, psnr_y, ssim_rgb, psnr_rgb


def process_pair(video_path, gt_image_path):
    try:
        gt = read_image(gt_image_path)
        vf = get_last_frame(video_path)
        return calc_metrics(vf, gt)
    except Exception as e:
        print(f"Error: {e}")
        return None


def evaluate_folder(video_root, image_root, save_dir, max_workers=8):

    os.makedirs(save_dir, exist_ok=True)
    results = []

    for root, dirs, files in os.walk(video_root):
        dirs.sort()
        videos = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        print(videos)
        if not videos:
            continue

        rel_folder = os.path.relpath(root, video_root)
        video_subfolder = rel_folder.split(os.sep)[-1]
        folder_results = []

        print(f"\nProcessing folder: {video_subfolder}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for v in sorted(videos):
                video_path = os.path.join(root, v)
                gt_folder_path = os.path.join(image_root, rel_folder)
                gt_image_path = os.path.join(gt_folder_path, os.path.splitext(v)[0] + ".png")

                if not os.path.exists(gt_image_path):
                    for ext in [".JPG", ".jpg"]:
                        alt = os.path.join(gt_folder_path, os.path.splitext(v)[0] + ext)
                        if os.path.exists(alt):
                            gt_image_path = alt
                            break
                    else:
                        print(f"GT not found for {v}")
                        continue

                futures.append(executor.submit(process_pair, video_path, gt_image_path))

            for f in tqdm(futures, desc=f"Evaluating {video_subfolder}", unit="video"):
                res = f.result()
                if res:
                    folder_results.append(res)

        if folder_results:
            folder_results = np.array(folder_results)

            mean_ssim_y = folder_results[:, 0].mean()
            mean_psnr_y = folder_results[:, 1].mean()
            mean_ssim_rgb = folder_results[:, 2].mean()
            mean_psnr_rgb = folder_results[:, 3].mean()

            results.append({
                "folder": video_subfolder,
                "mean_ssim_y": mean_ssim_y,
                "mean_psnr_y": mean_psnr_y,
                "mean_ssim_rgb": mean_ssim_rgb,
                "mean_psnr_rgb": mean_psnr_rgb,
                "video_count": len(folder_results)
            })

            save_path = os.path.join(save_dir, f"{video_subfolder}.csv")
            pd.DataFrame(
                folder_results,
                columns=["ssim_y", "psnr_y", "ssim_rgb", "psnr_rgb"]
            ).to_csv(save_path, index=False)

    if results:
        total_df = pd.DataFrame(results)
        total_videos = total_df["video_count"].sum()

        weighted_mean_ssim_y = (total_df["mean_ssim_y"] * total_df["video_count"]).sum() / total_videos
        weighted_mean_psnr_y = (total_df["mean_psnr_y"] * total_df["video_count"]).sum() / total_videos
        weighted_mean_ssim_rgb = (total_df["mean_ssim_rgb"] * total_df["video_count"]).sum() / total_videos
        weighted_mean_psnr_rgb = (total_df["mean_psnr_rgb"] * total_df["video_count"]).sum() / total_videos

        overall_row = pd.DataFrame([{
            "folder": "Overall",
            "mean_ssim_y": weighted_mean_ssim_y,
            "mean_psnr_y": weighted_mean_psnr_y,
            "mean_ssim_rgb": weighted_mean_ssim_rgb,
            "mean_psnr_rgb": weighted_mean_psnr_rgb,
            "video_count": total_videos
        }])

        total_df = pd.concat([total_df, overall_row], ignore_index=True)

        summary_path = os.path.join(save_dir, "summary.csv")
        total_df.to_csv(summary_path, index=False)

        xlsx_data = {
            row["folder"]: [
                row["mean_psnr_y"],
                row["mean_ssim_y"],
                row["mean_psnr_rgb"],
                row["mean_ssim_rgb"],
            ]
            for _, row in total_df.iterrows()
        }

        xlsx_df = pd.DataFrame(
            xlsx_data,
            index=["PSNR_Y", "SSIM_Y", "PSNR_RGB", "SSIM_RGB"]
        )

        xlsx_path = os.path.join(save_dir, "summary.xlsx")
        xlsx_df.to_excel(xlsx_path)

        print(f"\nDone.")
        print(f"Summary saved to: {os.path.abspath(summary_path)}")
        print(f"XLSX saved to: {os.path.abspath(xlsx_path)}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=128)
    args = parser.parse_args()

    evaluate_folder(args.video_root, args.image_root, args.save_dir, args.max_workers)
