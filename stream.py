import cv2
import numpy as np
import requests
import time
import threading
from PIL import Image, ImageSequence

# 表情对应的GIF文件路径
emotion_gifs = {
    'Angry': 'C:/Users/16503/Desktop/LocalInfre/angry.gif',
    'Disgust': 'C:/Users/16503/Desktop/LocalInfre/disgust.gif',
    'Fear': 'C:/Users/16503/Desktop/LocalInfre/fear.gif',
    'Happy': 'C:/Users/16503/Desktop/LocalInfre/happy.gif',
    'Sad': 'C:/Users/16503/Desktop/LocalInfre/sad.gif',
    'Surprise': 'C:/Users/16503/Desktop/LocalInfre/surprise.gif',
    'Neutral': 'C:/Users/16503/Desktop/LocalInfre/neutral.gif'
}

def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert('RGB')
        frames.append(frame)
    return frames

def display_gif(frames, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        for frame in frames:
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            cv2.imshow('GIF Emotion Display', frame_cv)
            if cv2.waitKey(int(duration * 10000 / len(frames))) & 0xFF == ord('q'):
                return
            # time.sleep(0.1)  # 每帧暂停0.1秒

def capture_and_send_video(url):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    prev_time = 0
    current_label = "No Result"
    current_inference_time = "N/A"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每秒推理一帧
        curr_time = time.time()
        
        if curr_time - prev_time >= 0.6:
            prev_time = curr_time

            # 保存当前帧到文件
            cv2.imwrite('C:/Users/16503/Desktop/LocalInfre/current_frame.jpg', frame)

            start_time = time.time()
            

            # 发送当前帧到服务器
            with open('C:/Users/16503/Desktop/LocalInfre/current_frame.jpg', 'rb') as f:
                files = {'image': f}
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    result = response.json()
                    if result:
                        current_label = result  # 更新结果
                        end_time = time.time()
                        current_inference_time = f"{(end_time - start_time):.2f} seconds"

                        if current_label in emotion_gifs: #显示表情
                            gif_frames = load_gif_frames(emotion_gifs[current_label])
                            display_gif(gif_frames, 0.5)


        # 显示当前推理结果和推理时间,实时视频流
        cv2.putText(frame, f'Emotion: {current_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference time: {current_inference_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = 'http://your_severID:5000/predict'  # 确保包含 http:// 协议
    capture_and_send_video(url)