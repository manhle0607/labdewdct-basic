import cv2
import numpy as np
import argparse
import os
import sys
import glob
import shutil
from scipy.fft import dct, idct

def dct_energy(block):
    """Tính năng lượng khối DCT (bỏ hệ số DC)."""
    return np.sum(block[1:] ** 2)

def text_to_bits(text):
    return ''.join(f'{ord(c):08b}' for c in text)

def bits_to_text(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

def embed_bits_in_frame(frame, bitstream, bit_per_block=1, block=8, threshold=10000):
    height, width = frame.shape[:2]
    bit_index = 0
    modified = 0

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(float)
    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            if bit_index >= len(bitstream):
                break

            patch = ycrcb[y:y+block, x:x+block, 0]
            patch_dct = dct(dct(patch.T, norm='ortho').T, norm='ortho')
            if dct_energy(patch_dct) > threshold:
                bits = bitstream[bit_index:bit_index+bit_per_block].ljust(bit_per_block, '0')
                for b, bit in enumerate(bits):
                    idx = (4 + b) % block
                    patch_dct[idx, idx] = (abs(patch_dct[idx, idx]) + 50) * (1 if bit == '1' else -1)

                patch_idct = idct(idct(patch_dct.T, norm='ortho').T, norm='ortho')
                ycrcb[y:y+block, x:x+block, 0] = np.clip(patch_idct, 0, 255)
                bit_index += bit_per_block
                modified += 1
        if bit_index >= len(bitstream):
            break

    output = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return output, bit_index == len(bitstream), bit_index, modified

def extract_bits_from_frame(frame, bit_limit, bit_per_block=1, block=8, threshold=10000):
    height, width = frame.shape[:2]
    bits = []
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(float)

    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            if len(bits) >= bit_limit:
                break
            patch = ycrcb[y:y+block, x:x+block, 0]
            patch_dct = dct(dct(patch.T, norm='ortho').T, norm='ortho')
            if dct_energy(patch_dct) > threshold:
                for b in range(bit_per_block):
                    idx = (4 + b) % block
                    bits.append('1' if patch_dct[idx, idx] > 5 else '0')
                    if len(bits) >= bit_limit:
                        break
        if len(bits) >= bit_limit:
            break

    return bits

def embed_message_batch(input_folder, message_file, output_folder, bits_per_block=1, block_size=8, threshold=10000):
    if not os.path.exists(input_folder):
        sys.exit(f"❌ Thư mục frames không tồn tại: {input_folder}")
    if not os.path.exists(message_file):
        sys.exit(f"❌ Không tìm thấy file thông điệp: {message_file}")

    if os.path.exists(output_folder):
        if input(f"⚠️ Thư mục '{output_folder}' đã tồn tại. Ghi đè? (y/n): ").lower() != 'y':
            sys.exit("✅ Đã huỷ thao tác.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    with open(message_file, 'r') as f:
        message_bits = text_to_bits(f.read())

    frame_paths = sorted(glob.glob(os.path.join(input_folder, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not frame_paths:
        sys.exit("❌ Không tìm thấy frame PNG.")

    print(f"[INFO] Đang nhúng thông điệp vào {len(frame_paths)} khung hình...")
    bit_index = 0
    modified_count = 0

    for idx, path in enumerate(frame_paths):
        img = cv2.imread(path)
        if bit_index < len(message_bits):
            img, done, used, changed = embed_bits_in_frame(img, message_bits[bit_index:], bits_per_block, block_size, threshold)
            bit_index += used
            modified_count += changed
            if done:
                print(f"[DONE] Nhúng xong trong {idx+1} frames. {modified_count} khối bị thay đổi.")
        output_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(output_path, img)

    if bit_index < len(message_bits):
        print(f"[⚠️] Thiếu khối đủ năng lượng! Nhúng {bit_index}/{len(message_bits)} bits.")
    else:
        print(f"[OK] Nhúng thành công toàn bộ thông điệp vào {output_folder}")

def recover_message_batch(frames_folder, output_file, char_count, bits_per_block=1, block_size=8, threshold=10000):
    if not os.path.exists(frames_folder):
        sys.exit(f"❌ Không tìm thấy thư mục: {frames_folder}")

    frames = sorted(glob.glob(os.path.join(frames_folder, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not frames:
        sys.exit("❌ Không có file PNG trong thư mục.")

    total_bits_needed = char_count * 8
    collected_bits = []

    print(f"[INFO] Trích xuất thông điệp từ {len(frames)} khung hình...")

    for path in frames:
        if len(collected_bits) >= total_bits_needed:
            break
        img = cv2.imread(path)
        bits = extract_bits_from_frame(img, total_bits_needed - len(collected_bits), bits_per_block, block_size, threshold)
        collected_bits.extend(bits)

    final_message = bits_to_text(''.join(collected_bits[:total_bits_needed]))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(final_message)
    print(f"[✅] Đã khôi phục thông điệp tại: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="DCT Video Frame Steganography")
    parser.add_argument("mode", choices=["hide", "recover"], help="Chế độ: hide (giấu tin) hoặc recover (trích xuất)")
    parser.add_argument("--frames-dir", default="output/frames", help="Thư mục ảnh đầu vào")
    parser.add_argument("--message", default="message.txt", help="File thông điệp")
    parser.add_argument("--output-dir", default="output/stego_frames", help="Thư mục đầu ra cho ảnh stego")
    parser.add_argument("--output", default="output/recovered.txt", help="File đầu ra cho thông điệp trích xuất")
    parser.add_argument("--num-bits", type=int, default=1, help="Số bit nhúng mỗi khối")
    parser.add_argument("--block-size", type=int, default=8, help="Kích thước khối DCT")
    parser.add_argument("--energy-threshold", type=float, default=10000, help="Ngưỡng năng lượng")
    parser.add_argument("--msg-length", type=int, default=23, help="Số ký tự thông điệp cần trích xuất")

    args = parser.parse_args()

    if args.mode == "hide":
        embed_message_batch(args.frames_dir, args.message, args.output_dir, args.num_bits, args.block_size, args.energy_threshold)
    elif args.mode == "recover":
        recover_message_batch(args.frames_dir, args.output, args.msg_length, args.num_bits, args.block_size, args.energy_threshold)

if __name__ == "__main__":
    main()

