import cv2
import pyaudio
import wave
import threading
import time
import os
import numpy as np
from datetime import datetime
import speech_recognition as sr


import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i).get('name'))


# ==== Setup ====
user = input("Input user name: ")
save_dir = f"wild_dataset/{user}/database"
os.makedirs(save_dir, exist_ok=True)

url = "http://10.181.123.162:4747/video"
cap = cv2.VideoCapture(url)

audio_format = pyaudio.paInt16
channels = 1
rate = 44100
chunk = 1024
record_seconds = 5

p = pyaudio.PyAudio()

# Try to open audio stream
try:
    audio_stream = p.open(format=audio_format,
                          channels=channels,
                          rate=rate,
                          input=True,
                          frames_per_buffer=chunk)
    audio_ok = True
except Exception as e:
    print(f"[!] Audio stream fallback: {e}")
    audio_ok = False

current_frame = None
frame_lock = threading.Lock()


# ==== Frame Capture ====
def save_frame(label):
    with frame_lock:
        if current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, current_frame)
            print(f"[+] Saved frame: {filename}")


def capture_video():
    global current_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[-] Failed to capture video frame.")
            break
        with frame_lock:
            current_frame = frame
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ==== Audio Chunk Capture ====
def capture_audio_chunk(label):
    if not audio_ok:
        print("[!] Audio capture skipped (fallback active)")
        return
    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = audio_stream.read(chunk)
        frames.append(data)

    filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    filepath = os.path.join(save_dir, filename)
    # with wave.open(filepath, 'wb') as wf:
    #     wf.setnchannels(channels)
    #     wf.setsampwidth(p.get_sample_size(audio_format))
    #     wf.setframerate(rate)
    #     wf.writeframes(b''.join(frames))
    #print(f"[+] Saved audio: {filename}")


# ==== Speech Recognition ====
def listen_for_command():
    recognizer = sr.Recognizer()
    mic_source = sr.Microphone()
    while True:
        try:
            with mic_source as source:
                print("Listening for wake-up word...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            command = recognizer.recognize_google(audio).lower()
            print(f"Heard: {command}")
            if "hello" in command:
                label = command.replace("hello", "").strip()
                if label:
                    print(f"[✓] Triggered label: '{label}'")
                    save_frame(label)
                    capture_audio_chunk(label)
        except Exception as e:
            print(f"[!] Speech error: {e}")


# ==== Run All Threads ====
def main():
    threading.Thread(target=capture_video, daemon=True).start()
    threading.Thread(target=listen_for_command, daemon=True).start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
