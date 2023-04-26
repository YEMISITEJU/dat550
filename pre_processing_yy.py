import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import time

data_dir = '/Volumes/storage/dfdc_preview_set'
processed_data_dir = os.path.join(data_dir, "processed_data")
os.makedirs(processed_data_dir, exist_ok=True)
metadata_path = os.path.join(data_dir, 'dataset.json')

# Load the metadata
metadata = json.load(open(metadata_path))

# filter out videos not in the "train" split
train_videos = [filename for filename, info in metadata.items() if info["set"] == "train"]

# Define augmentation parameters
rotation_angles = [-10, -5, 0, 5, 10]  # list of rotation angles to randomly choose from
horizontal_flips = [True, False]  # list of boolean values to randomly choose from

# Define face detection parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
scale_factor = 2  # factor to scale up the bounding box of the face

extraction_rate = 5  # desired frame for extraction

total_processing_time = 0
num_processed_videos = 0

# Count the number of mp4 files in the train folder
num_train_videos = len(train_videos)
print("Number of training videos: ", num_train_videos)

def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    else:
        # select the largest face in the frame
        largest_face = max(faces, key=lambda x: x[2])
        x, y, w, h = largest_face
        x -= w // 2
        y -= h // 2
        x = max(0, x)
        y = max(0, y)
        return frame[y:y + h * scale_factor, x:x + w * scale_factor]
    
def augment_frames(frame):
    angle = np.random.choice(rotation_angles)
    flip = np.random.choice(horizontal_flips)
    M = cv2.getRotationMatrix2D((256/2, 256/2), angle, 1.0)
    frame = cv2.warpAffine(frame, M, (256, 256))
    if flip:
        frame = cv2.flip(frame, 1)
    return frame

def resized_frames(frame, size=256):
    return cv2.resize(frame, (size, size))

def normalize_frames(frames):
    mean = np.mean(frames, axis=(0,1,2)) / 255
    std = np.std(frames, axis=(0,1,2)) / 255
    frame = (frames / 255 - mean) / std
    return frame

for video_filename in train_videos:
    start_time = time.time()
    npz_filename = os.path.join(processed_data_dir, f"{video_filename[:-4]}.npz")
    if os.path.exists(npz_filename):
        continue
    video_path = os.path.join(data_dir, video_filename)
    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # get actual frame rate of video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get total number of frames in video
    desired_num_frames = int(np.ceil(num_frames / fps * extraction_rate))  # calculate desired number of frames
    frames = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # resize cropped face frame to desired dimensions
        #frame = resized_frames(frame)
        
        # detect face and scale up bounding box
        face_frame = face_detection(frame)
        if face_frame is None:
            continue

        # resize cropped face frame to desired dimensions
        face_frame = resized_frames(face_frame)

        # apply random augmentations
        face_frame = augment_frames(face_frame)

        frames.append(face_frame)
        if len(frames) >= desired_num_frames:
            break
    cap.release()

    if len(frames) == 0:
        continue
    
    frames = np.stack(frames, axis=0)

    # normalize pixel values to range [0, 1]
    frames = normalize_frames(frames)
    
    npz_dirname = os.path.dirname(npz_filename)
    os.makedirs(npz_dirname, exist_ok=True)
    np.savez_compressed(npz_filename, frames)

    end_time = time.time()
    processing_time = end_time - start_time
    total_processing_time += processing_time
    num_processed_videos += 1
    num_train_videos -= 1

    # calculate average processing time per video
    avg_processing_time = total_processing_time / num_processed_videos
    
    # calculate estimated time remaining
    time_remaining = avg_processing_time * num_train_videos

    # convert time remaining to hours, minutes, seconds
    hours, remainder = divmod(time_remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Estimated time remaining: {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds")
    print(f"{num_train_videos} videos remaining")
    print()
   
