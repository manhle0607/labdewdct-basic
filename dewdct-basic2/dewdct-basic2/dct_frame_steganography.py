import cv2
import numpy as np
import os
import sys
import glob
import shutil
import argparse
from scipy.fft import dct, idct

def dct_energy(block):
    """Tính năng lượng của khối DCT, loại bỏ hệ số DC."""
    return np.sum(block[1:] ** 2)

def text_to_bits(text):
    """Chuyển chuỗi ký tự sang dãy bit nhị phân."""
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    """Chuyển chuỗi bit nhị phân thành chuỗi ký tự."""
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def embed_bits_in_frame(frame, bitstream, block_size=8, threshold=10000, bits_per_block=1):
    """Nhúng bit vào 1 khung hình sử dụng DCT và ngưỡng năng lượng."""
    height, width = frame.shape[:2]
    bit_index = 0
    modified_blocks = 0

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(float)

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            if bit_index >= len(bitstream):
                break

            patch = ycrcb[y:y+block_size, x:x+block_size, 0]
            patch_dct = dct(dct(patch.T, norm='ortho').T, norm='ortho')
            if dct_energy(patch_dct) > threshold:
                bits = bitstream[bit_index:bit_index+bits_per_block].ljust(bits_per_block, '0')
                for k, b in enumerate(bits):
                    idx = (4 + k) % block_size
                    patch_dct[idx, idx] = (abs(patch_dct[idx, idx]) + 50) * (1 if b == '1' else -1)
                patch_idct = idct(idct(patch_dct.T, norm='ortho').T, norm='ortho')
                ycrcb[y:y+block_size, x:x+block_size, 0] = np.clip(patch_idct, 0, 255)
                bit_index += bits_per_block
                modified_blocks += 1

        if bit_index >= len(bitstream):
            break

    output_frame = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return output_frame, bit_index == len(bitstream), bit_index, modified_blocks

def extract_bits_from_frame(frame, max_bits, block_size=8, threshold=10000, bits_per_block=1):
    """Trích xuất bit từ 1 khung hình đã giấu tin."""
    height, width = frame.shape[:2]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    bits = []
    count = 0

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            if count >= max_bits:
                break
            patch = ycrcb[y:y+block_size, x:x+block_size, 0].astype(float)
            patch_dct = dct(dct(patch.T, norm='ortho').T, norm='ortho')
            if dct_energy(patch_dct) > threshold:
                for k in range(bits_per_block):
                    idx = (4 + k) % block_size
                    bit = '1' if patch_dct[idx, idx] > 5 else '0'
                    bits.append(bit)
                    count += 1
                    if count >= max_bits:
                        break
        if count >= max_bits:
            break

    return bits

def embed_message_to_frames(frames_dir, message_path, output_dir, bits_per_block=1, block_size=8, threshold=10000):
    """Nhúng thông điệp vào nhiều khung hình và lưu lại."""
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {frames_dir}")
    if not os.path.isfile(message_path):
        raise FileNotFoundError(f"Không tìm thấy file thông điệp: {message_path}")

    if os.path.exists(output_dir):
        confirm = input(f"⚠️ Thư mục '{output_dir}' đã tồn tại. Ghi đè? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❎ Đã huỷ thao tác.")
            sys.exit(1)
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(message_path, 'r') as f:
        message = f.read()
    bitstream = text_to_bits(message)

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not frame_files:
        raise ValueError("❌ Không có file PNG trong thư mục.")

    bit_index = 0
    modified_blocks = 0

    for i, frame_path in enumerate(frame_files):
        img = cv2.imread(frame_path)
        if img is None:
            print(f"[!] Không đọc được khung hình '{frame_path}', sẽ sao chép nguyên.")
            shutil.copy(frame_path, os.path.join(output_dir, os.path.basename(frame_path)))
            continue

        if bit_index < len(bitstream):
            img, done, bits_written, changed = embed_bits_in_frame(img, bitstream[bit_index:], block_size, threshold, bits_per_block)
            bit_index += bits_written
            modified_blocks += changed
            if done:
                print(f"[OK] Nhúng xong sau {i+1} frames với {modified_blocks} khối thay đổi.")
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, img)

    if bit_index < len(bitstream):
        print(f"[⚠️] Không đủ khối có năng lượng cao. Nhúng {bit_index}/{len(bitstream)} bits.")
    else:
        print(f"[✅] Nhúng hoàn tất vào {output_dir}")

def recover_message_from_frames(frames_dir, output_file, char_count, bits_per_block=1, block_size=8, threshold=10000):
    """Trích xuất thông điệp từ các khung hình và ghi vào file."""
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {frames_dir}")

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not frame_files:
        raise ValueError("Không tìm thấy khung hình PNG.")

    total_bits = char_count * 8
    collected_bits = []

    for path in frame_files:
        if len(collected_bits) >= total_bits:
            break
        img = cv2.imread(path)
        if img is None:
            print(f"[!] Bỏ qua ảnh lỗi: {path}")
            continue
        bits = extract_bits_from_frame(img, total_bits - len(collected_bits), block_size, threshold, bits_per_block)
        collected_bits.extend(bits)

    message = bits_to_text(''.join(collected_bits[:total_bits]))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(message)
    print(f"[✅] Đã khôi phục thông điệp tại: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Giấu & Trích xuất thông điệp bằng DCT trên khung hình")
    parser.add_argument("mode", choices=["hide", "recover"], help="'hide' để nhúng, 'recover' để trích")
    parser.add_argument("--frames-dir", default="output/frames", help="Thư mục chứa khung hình PNG")
    parser.add_argument("--message", default="message.txt", help="File chứa thông điệp cần giấu")
    parser.add_argument("--output-dir", default="output/stego_frames", help="Thư mục lưu khung hình đã giấu")
    parser.add_argument("--output", default="output/recovered.txt", help="Tệp chứa thông điệp đã khôi phục")
    parser.add_argument("--num-bits", type=int, default=1, help="Số bit giấu mỗi khối")
    parser.add_argument("--block-size", type=int, default=8, help="Kích thước khối DCT")
    parser.add_argument("--energy-threshold", type=float, default=10000, help="Ngưỡng năng lượng để chọn khối")
    parser.add_argument("--msg-length", type=int, default=23, help="Độ dài thông điệp cần khôi phục (ký tự)")

    args = parser.parse_args()

    if args.mode == "hide":
        embed_message_to_frames(args.frames_dir, args.message, args.output_dir,
                                bits_per_block=args.num_bits,
                                block_size=args.block_size,
                                threshold=args.energy_threshold)
    elif args.mode == "recover":
        recover_message_from_frames(args.frames_dir, args.output,
                                    char_count=args.msg_length,
                                    bits_per_block=args.num_bits,
                                    block_size=args.block_size,
                                    threshold=args.energy_threshold)

if __name__ == "__main__":
    main()

