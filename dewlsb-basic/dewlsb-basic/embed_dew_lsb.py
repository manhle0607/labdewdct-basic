import cv2
import numpy as np
import os
import argparse
import glob
from scipy.fft import dct

def compute_energy(block):
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return np.sum(dct_block[1:]**2)

def text_to_bits(text):
    return ''.join(f'{ord(c):08b}' for c in text)

def embed_lsb(frame, bits, block_size=8, threshold=10000):
    h, w = frame.shape[:2]
    idx = 0
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0]

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if idx >= len(bits):
                break
            block = Y[i:i+block_size, j:j+block_size].astype(float)
            if compute_energy(block) > threshold:
                cx, cy = j + block_size//2, i + block_size//2
                Y[cy, cx] = (int(Y[cy, cx]) & ~1) | int(bits[idx])
                idx += 1
        if idx >= len(bits):
            break

    ycc[:, :, 0] = Y
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="output/frames", help="Directory of input frames")
    parser.add_argument("--message", default="message.txt", help="Message file to hide")
    parser.add_argument("--output-dir", default="output/stego_frames", help="Output directory for stego frames")
    parser.add_argument("--threshold", type=float, default=10000, help="Energy threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.message, 'r') as f:
        message = f.read()
    bits = text_to_bits(message)

    files = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    bit_idx = 0

    for i, file in enumerate(files):
        frame = cv2.imread(file)
        if bit_idx < len(bits):
            frame = embed_lsb(frame, bits[bit_idx:], threshold=args.threshold)
            # Không cần tính số bit thực sự nhúng ở đây vì dùng max energy theo khối
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(file)), frame)

    print(f"✅ Embedding complete. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()

