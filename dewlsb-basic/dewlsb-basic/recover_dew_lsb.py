import cv2
import numpy as np
import os
import argparse
import glob
from scipy.fft import dct

def compute_energy(block):
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return np.sum(dct_block[1:]**2)

def bits_to_text(bits):
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

def extract_lsb(frame, num_bits, block_size=8, threshold=10000):
    h, w = frame.shape[:2]
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0]
    bits = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if len(bits) >= num_bits:
                break
            block = Y[i:i+block_size, j:j+block_size].astype(float)
            if compute_energy(block) > threshold:
                cx, cy = j + block_size//2, i + block_size//2
                bits.append(str(int(Y[cy, cx]) & 1))
        if len(bits) >= num_bits:
            break
    return ''.join(bits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="output/stego_frames", help="Directory of stego frames")
    parser.add_argument("--output", default="output/recovered.txt", help="Output recovered message file")
    parser.add_argument("--msg-length", type=int, default=23, help="Length of hidden message")
    parser.add_argument("--threshold", type=float, default=10000, help="Energy threshold")
    args = parser.parse_args()

    bits_needed = args.msg_length * 8
    #files = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    files = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("frame_", "")))


    bits = []
    for file in files:
        if len(bits) >= bits_needed:
            break
        frame = cv2.imread(file)
        bits += list(extract_lsb(frame, bits_needed - len(bits), threshold=args.threshold))

    message = bits_to_text(''.join(bits[:bits_needed]))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(message)
    print(f"✅ Message recovered to {args.output}")

if __name__ == "__main__":
    main()

