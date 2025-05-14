import cv2
import numpy as np
import os
import sys
import glob
import argparse
from tqdm import tqdm
from scipy.fft import dct, idct

def check_dependencies():
    """Kiểm tra các thư viện cần thiết."""
    try:
        import cv2
        import numpy
        import scipy.fft
        import tqdm
    except ImportError as e:
        print(f"[ERROR] Thiếu thư viện: {e.name}. Cài bằng: pip3 install {e.name.lower()}")
        sys.exit(1)

def dct_energy(block):
    """Tính năng lượng của khối DCT, bỏ hệ số DC."""
    return np.sum(block[1:] ** 2)

def bits_to_text(binary):
    """Chuyển dãy bit nhị phân thành chuỗi ký tự."""
    if len(binary) < 8:
        return ""
    chars = []
    for i in range(0, len(binary) - 7, 8):
        try:
            chars.append(chr(int(binary[i:i+8], 2)))
        except ValueError:
            break
    return ''.join(chars)

def extract_bits_from_frame(frame, bit_limit, block_size=8, threshold=10000, bits_per_block=1):
    """Trích xuất bit từ khung hình dựa vào các khối có năng lượng cao."""
    height, width = frame.shape[:2]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    extracted_bits = []
    count = 0

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            if count >= bit_limit:
                break
            block = ycrcb[y:y+block_size, x:x+block_size, 0].astype(float)
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            if dct_energy(dct_block) > threshold:
                for k in range(bits_per_block):
                    idx = (4 + k) % block_size
                    bit = '1' if dct_block[idx, idx] > 0 else '0'
                    extracted_bits.append(bit)
                    count += 1
                    if count >= bit_limit:
                        break
        if count >= bit_limit:
            break

    return extracted_bits

def recover_message_from_frames(frames_path, output_path, message_length, reference_path=None,
                                 bits_per_block=1, block_size=8, threshold=10000):
    """Trích xuất thông điệp từ tất cả khung hình PNG trong thư mục."""
    if not os.path.isdir(frames_path):
        raise FileNotFoundError(f"❌ Thư mục '{frames_path}' không tồn tại.")

    frame_list = sorted(glob.glob(os.path.join(frames_path, "*.png")),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not frame_list:
        raise ValueError("❌ Không tìm thấy file PNG trong thư mục.")

    print(f"[INFO] Trích xuất từ {len(frame_list)} khung hình...")

    total_bits = message_length * 8
    bitstream = []

    for frame_file in tqdm(frame_list, desc="🧩 Đang xử lý"):
        if len(bitstream) >= total_bits:
            break
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"[!] Bỏ qua frame lỗi: {frame_file}")
            continue
        bits = extract_bits_from_frame(frame, total_bits - len(bitstream),
                                       block_size=block_size,
                                       threshold=threshold,
                                       bits_per_block=bits_per_block)
        bitstream.extend(bits)

    if len(bitstream) < total_bits:
        print(f"[⚠️] Chỉ trích xuất được {len(bitstream)} bit (cần {total_bits}).")

    recovered = bits_to_text(''.join(bitstream[:total_bits]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out:
        out.write(recovered)

    print(f"[✅] Đã lưu thông điệp khôi phục tại: {output_path}")

    # So sánh với thông điệp gốc nếu có
    if reference_path and os.path.exists(reference_path):
        with open(reference_path, 'r') as ref:
            original = ref.read()
        if recovered == original:
            print("🎉 Thành công: Thông điệp trích ra khớp với thông điệp gốc.")
        else:
            print("❗ Thông điệp trích ra KHÔNG khớp với thông điệp gốc.")
            print(f"[Extracted] {recovered}")
            print(f"[Original ] {original}")

def main():
    check_dependencies()

    parser = argparse.ArgumentParser(description="Trích xuất thông điệp ẩn trong khung hình PNG sử dụng DCT")
    parser.add_argument("--frames-dir", default="output/stego_frames", help="Thư mục chứa ảnh PNG đã giấu tin")
    parser.add_argument("--output", default="output/recovered.txt", help="File đầu ra chứa thông điệp trích xuất")
    parser.add_argument("--original-message", default="message.txt", help="File thông điệp gốc (tùy chọn để so sánh)")
    parser.add_argument("--msg-length", type=int, default=23, help="Số ký tự cần khôi phục (mặc định: 23)")
    parser.add_argument("--num-bits", type=int, default=1, help="Số bit đã nhúng mỗi khối")
    parser.add_argument("--block-size", type=int, default=8, help="Kích thước khối DCT")
    parser.add_argument("--energy-threshold", type=float, default=10000, help="Ngưỡng năng lượng để chọn khối")

    args = parser.parse_args()

    try:
        recover_message_from_frames(args.frames_dir, args.output,
                                    message_length=args.msg_length,
                                    reference_path=args.original_message,
                                    bits_per_block=args.num_bits,
                                    block_size=args.block_size,
                                    threshold=args.energy_threshold)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

