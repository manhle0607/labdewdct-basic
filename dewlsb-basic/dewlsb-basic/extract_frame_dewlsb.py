import cv2
import os
import argparse
from tqdm import tqdm
from PIL import Image

def extract_frames(video_path, output_dir, max_frames=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames, max_frames) if max_frames else total_frames

    for i in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(img).save(os.path.join(output_dir, f"{i}.png"))
    cap.release()
    print(f"✅ Saved {total_frames} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="input.avi", help="Input video file")
    parser.add_argument("--output", default="output/frames", help="Output frames directory")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames")
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.max_frames)

