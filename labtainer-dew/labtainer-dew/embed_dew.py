import cv2
import numpy as np
import os

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def embed_bit(blockA, blockB, bit):
    EA = np.sum(np.square(dct2(blockA)))
    EB = np.sum(np.square(dct2(blockB)))
    if bit == 1 and EA <= EB:
        blockA += 1
    elif bit == 0 and EA >= EB:
        blockB += 1
    return blockA, blockB

#    ^=^t    B            ^zC 1: Th      ng    ^qi      ^gp c         n gi         u
message = "Hello DEW steganography!"
bitstream = []

#    ^=^t^a B            ^zC 2: Chuy      ^cn k       t          th      nh bit
for char in message:
    bits = format(ord(char), '08b')  # chuy      ^cn m      ^wi k       t          th      nh 8 bit
    bitstream.extend([int(b) for b in bits])

bit_idx = 0

os.makedirs('stego_frames', exist_ok=True)
frame_files = sorted(os.listdir('frames'))

for fname in frame_files:
    path = os.path.join('frames', fname)
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.shape[0] >= 8 and gray.shape[1] >= 16 and bit_idx < len(bitstream):
        blockA = gray[0:8, 0:8].copy()
        blockB = gray[0:8, 8:16].copy()
        newA, newB = embed_bit(blockA, blockB, bitstream[bit_idx])
        gray[0:8, 0:8] = newA
        gray[0:8, 8:16] = newB
        bit_idx += 1

    stego_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f'stego_frames/{fname}', stego_frame)

print(f"   ^|^e    ^p       nh      ng {bit_idx} bit (t            ng    ^q            ng {bit_idx//8} k       t         ) v      o video.")
