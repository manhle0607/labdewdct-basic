import cv2
import os

out = cv2.VideoWriter('output_stego.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (360, 640))
frame_files = sorted(os.listdir('stego_frames'))

for fname in frame_files:
    path = os.path.join('stego_frames', fname)
    frame = cv2.imread(path)
    out.write(frame)

out.release()
print("   ^|^e    ^p       t         o video ch         a tin: output_stego.avi")
