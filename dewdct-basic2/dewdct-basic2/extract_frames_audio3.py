import cv2
import os
import sys
import shutil
import argparse
from tqdm import tqdm
from PIL import Image

def verify_dependencies():
    """Kiểm tra xem các thư viện cần thiết đã được cài đặt chưa."""
    try:
        import cv2
        import tqdm
        import PIL
    except ImportError as lib_error:
        print(f"[ERROR] Thiếu thư viện: {lib_error.name}. Cài đặt bằng: pip3 install {lib_error.name.lower()}")
        sys.exit(1)

def prepare_output_folder(folder_path):
    """Tạo lại thư mục đầu ra nếu người dùng đồng ý ghi đè."""
    if os.path.exists(folder_path):
        choice = input(f"⚠️ Thư mục '{folder_path}' đã tồn tại. Ghi đè? (y/n): ").strip().lower()
        if choice != 'y':
            print("🚫 Đã huỷ thao tác.")
            return False
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    return True

def extract_video_frames(video_file, save_folder, max_count=None):
    """Tách các khung hình từ video và lưu dưới dạng PNG."""
    if not os.path.isfile(video_file):
        print(f"[ERROR] Không tìm thấy video: {video_file}")
        return False

    if not prepare_output_folder(save_folder):
        return False

    capture = cv2.VideoCapture(video_file)
    if not capture.isOpened():
        print(f"[ERROR] Không thể mở video: {video_file}")
        return False

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_count:
        total = min(total, max_count)

    for idx in tqdm(range(total), desc="📸 Đang tách khung hình"):
        ret, frame = capture.read()
        if not ret:
            print(f"[WARNING] Không đọc được frame thứ {idx}.")
            break
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_path = os.path.join(save_folder, f"{idx}.png")
        Image.fromarray(rgb_img).save(output_path)

    capture.release()
    print(f"✅ Đã lưu khung hình tại: {save_folder}")
    return True

def main():
    verify_dependencies()

    parser = argparse.ArgumentParser(description="Tách khung hình từ video (bỏ âm thanh)")
    parser.add_argument("--video", default="input.avi", help="Tệp video đầu vào (mặc định: input.avi)")
    parser.add_argument("--output", default="output/frames", help="Thư mục lưu khung hình (mặc định: output/frames)")
    parser.add_argument("--limit", type=int, help="Số lượng khung hình tối đa cần tách")

    args = parser.parse_args()

    success = extract_video_frames(args.video, args.output, args.limit)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

