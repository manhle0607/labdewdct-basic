import cv2
import numpy as np

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def extract_bit(blockA, blockB):
    EA = np.sum(np.square(dct2(blockA)))
    EB = np.sum(np.square(dct2(blockB)))
    return 1 if EA > EB else 0

cap = cv2.VideoCapture('output_stego.avi')
bitstream = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] >= 8 and gray.shape[1] >= 16:
        blockA = gray[0:8, 0:8]
        blockB = gray[0:8, 8:16]
        bitstream.append(extract_bit(blockA, blockB))

cap.release()

#    ^=^t^d Gh      p l         i th      nh chu      ^wi k       t
chars = []
for i in range(0, len(bitstream), 8):
    byte = bitstream[i:i+8]
    if len(byte) < 8:
        break
    char = chr(int("".join(str(b) for b in byte), 2))
    chars.append(char)

message = ''.join(chars)
print("   ^=^s    Th      ng    ^qi      ^gp tr      ch xu         t:", message)
