import cv2
import numpy as np
import json

# simple face detection using cv2 (can be changed to pre-trained YOLO for better/faster results)
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

# simple data augmentation: horizontal flip and rotation    
def augment_frames(frame):
    angle = np.random.choice(rotation_angles)
    flip = np.random.choice(horizontal_flips)
    M = cv2.getRotationMatrix2D((256/2, 256/2), angle, 1.0)
    frame = cv2.warpAffine(frame, M, (256, 256))
    if flip:
        frame = cv2.flip(frame, 1)
    return frame

# resize frames to desired dimensions
def resized_frames(frame, size=256):
    return cv2.resize(frame, (size, size))

# normalize pixel values to range [0, 1]
def normalize_frames(frames):
    mean = np.mean(frames, axis=(0,1,2)) / 255
    std = np.std(frames, axis=(0,1,2)) / 255
    frame = (frames / 255 - mean) / std
    return frame



# Path to the dataset
data_dir = '/Users/yuriyyurchenko/Documents/UiS/Semester_4/DataMining/Project/dfdc_train_part_10/'
metadata_path = data_dir + 'metadata.json'
metadata = json.load(open(metadata_path)) # load metadata

# filter out videos not in the "train" split
train_videos = [filename for filename, info in metadata.items() if info["split"] == "train"]
train_videos = train_videos[:5] # using 5 for testing

# Define augmentation parameters
rotation_angles = [-10, -5, 0, 5, 10]  # rotation angles to randomly choose from
horizontal_flips = [True, False]  # list of boolean values to randomly choose from

# Define face detection parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
scale_factor = 2  # factor to scale up the bounding box of the face

extraction_rate = 1  # desired frame for extraction
    
for video_filename in train_videos:
    video_path = f"{data_dir}/{video_filename}"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # get actual frame rate of video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get total number of frames in video
    desired_num_frames = int(np.ceil(num_frames / fps * extraction_rate))  # calculate desired number of frames
    frames = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

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
    frames = np.stack(frames, axis=0)

    # normalize pixel values to range [0, 1]
    frames = normalize_frames(frames)

    # save pre-processed frames to disk
    np.save(f"{data_dir}/{video_filename[:-4]}.npy", frames)
