import subprocess
import argparse
import os
import sys
import glob

def check_ffmpeg():
    """Kiểm tra xem FFmpeg có được cài đặt không."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg chưa được cài đặt. Cài bằng: sudo apt install ffmpeg")
        sys.exit(1)

def confirm_output(output_path):
    """Kiểm tra tệp đầu ra, xác nhận nếu cần ghi đè."""
    if os.path.exists(output_path):
        response = input(f"⚠️ Tệp '{output_path}' đã tồn tại. Ghi đè? (y/n): ").strip().lower()
        if response != 'y':
            print("❎ Đã huỷ thao tác.")
            return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return True

def create_video_from_frames(frames_dir, output_file, fps=30):
    """Tạo video từ chuỗi khung hình PNG."""
    if not os.path.exists(frames_dir):
        print(f"❌ Không tìm thấy thư mục khung hình: {frames_dir}")
        return False

    if not confirm_output(output_file):
        return False

    # Kiểm tra danh sách khung hình
    #frame_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    frame_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("frame_", "")))

    if not frame_list:
        print(f"❌ Không có khung hình PNG nào trong thư mục: {frames_dir}")
        return False

    # Gọi ffmpeg để ghép khung hình
    try:
        subprocess.run([
            "ffmpeg",
            "-framerate", str(fps),
            "-i", f"{frames_dir}/%d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y", output_file
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        print(f"✅ video been created: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi tạo video: {e.stderr}")
        return False

def main():
    check_ffmpeg()

    parser = argparse.ArgumentParser(description="Tạo video từ chuỗi khung hình PNG")
    parser.add_argument("--frames-dir", default="output/stego_frames", help="Thư mục chứa khung hình (mặc định: output/stego_frames)")
    parser.add_argument("--output", default="output/combined_video.mp4", help="Tệp video đầu ra (mặc định: output/combined_video.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Tốc độ khung hình (mặc định: 30)")

    args = parser.parse_args()

    if not create_video_from_frames(args.frames_dir, args.output, args.fps):
        sys.exit(1)

if __name__ == "__main__":
    main()

