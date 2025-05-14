import os
import argparse
import subprocess
import glob

def combine_frames_to_video(frames_dir, output_path, fps=30):
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory '{frames_dir}' not found.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_file = os.path.join(frames_dir, "frame_%05d.png")
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    for idx, f in enumerate(files):
        os.rename(f, os.path.join(frames_dir, f"frame_{idx:05d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", temp_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ], check=True)
    print(f"✅ Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="output/stego_frames", help="Directory of frames")
    parser.add_argument("--output", default="output/stego_video.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()
    combine_frames_to_video(args.frames_dir, args.output, args.fps)

