import cv2
import os
import argparse
from tqdm import tqdm
import sys
import shutil
from PIL import Image

def check_required_libraries():
    try:
        import cv2
        import tqdm
        import PIL
    except ImportError as err:
        print(f"Thiếu thư viện: {err.name}. Cài bằng: pip3 install {err.name.lower()}")
        sys.exit(1)

def create_output_folder(folder_path):
    if os.path.exists(folder_path):
        response = input(f"Thư mục '{folder_path}' đã tồn tại. Ghi đè? (y/n): ").strip().lower()
        if response != 'y':
            print("Đã hủy thao tác.")
            return False
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    return True

def extract_video_frames(video_path, output_folder, frame_limit=None):
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy video '{video_path}'")
        return False

    if not create_output_folder(output_folder):
        return False

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Lỗi: Không thể mở video '{video_path}'")
        return False

    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_limit is not None:
        total = min(total, frame_limit)

    for frame_index in tqdm(range(total), desc="Tách khung hình"):
        ret, frame = video_capture.read()
        if not ret:
            print(f"Cảnh báo: Không đọc được frame {frame_index}.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img.save(os.path.join(output_folder, f"{frame_index}.png"))

    video_capture.release()
    print(f"✅ Đã lưu {frame_index + 1} khung hình tại: {output_folder}")
    return True

def main():
    check_required_libraries()

    parser = argparse.ArgumentParser(description="Tách khung hình từ video")
    parser.add_argument("--video", default="input.avi", help="Tệp video đầu vào (mặc định: input.avi)")
    parser.add_argument("--frames-dir", default="output/frames", help="Thư mục lưu ảnh (mặc định: output/frames)")
    parser.add_argument("--max-frames", type=int, help="Giới hạn số lượng khung hình cần tách")

    args = parser.parse_args()

    if not extract_video_frames(args.video, args.frames_dir, args.max_frames):
        sys.exit(1)

if __name__ == "__main__":
    main()

