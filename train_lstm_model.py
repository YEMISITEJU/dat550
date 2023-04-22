import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


data_dir = '/Volumes/storage/dfdc_preview_set'
processed_data_dir = os.path.join(data_dir, "processed_data")
metadata_path = os.path.join(data_dir, 'dataset.json')


# Load the metadata
with open(metadata_path) as f:
    metadata = json.load(f)

train_videos = [filename for filename, info in metadata.items() if info["set"] == "train"]

# Create a list of processed video paths and corresponding labels
processed_video_paths = []
labels = []

for video_filename in train_videos:
    npz_filename = os.path.join(processed_data_dir, f"{video_filename[:-4]}.npz")
    if os.path.exists(npz_filename):
        processed_video_paths.append(npz_filename)
        labels.append(1 if metadata[video_filename]['label'] == 'fake' else 0)


# Data generator
def data_generator(video_paths, labels, sequence_length, batch_size=4):
    num_videos = len(video_paths)
    height, width, channels = 256, 256, 3
    expected_features = height * width * channels
    
    while True:
        # Shuffle the data
        indices = np.random.permutation(np.arange(num_videos))
        video_paths = [video_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

        # Generate data in batches
        for i in range(0, num_videos, batch_size):
            batch_video_paths = video_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batch_data = []
            batch_labels_filtered = []
            for idx, video_path in enumerate(batch_video_paths):
                with np.load(video_path) as data:
                    video_data = data['arr_0']
                    # Pad the video data if the sequence length is less than the required sequence_length
                    if video_data.shape[0] < sequence_length:
                        padding = sequence_length - video_data.shape[0]
                        video_data = np.pad(video_data, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant')
                    video_data = video_data.reshape(sequence_length, -1)
                    if video_data.shape[1] == expected_features:  # Check if the video data shape is consistent
                        batch_data.append(video_data)
                        batch_labels_filtered.append(batch_labels[idx])

            if len(batch_data) == batch_size:  # Only yield complete batches
                batch_data = np.array(batch_data)
                batch_labels_filtered = np.array(batch_labels_filtered)

                yield batch_data, batch_labels_filtered




# Split the data into training, validation, and test sets
train_paths, temp_paths, train_labels, temp_labels = train_test_split(processed_video_paths, labels, test_size=0.4, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)

batch_size = 4
sequence_length = 76
num_features = 256 * 256 * 3

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, num_features), return_sequences=True),
    Dropout(0.4),
    LSTM(32, return_sequences=False),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
# Compile the model
optimizer = tf.keras.optimizers.legacy.Adam(lr=0.001)

# Define precision, recall, and f1 score metrics
precision = tf.keras.metrics.Precision(name='precision')
recall = tf.keras.metrics.Recall(name='recall')

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
train_steps = len(train_paths) // batch_size
val_steps = len(val_paths) // batch_size

train_gen = data_generator(train_paths, train_labels, sequence_length, batch_size)
val_gen = data_generator(val_paths, val_labels, sequence_length, batch_size)

model.fit(train_gen, steps_per_epoch=train_steps, validation_data=val_gen, validation_steps=val_steps, epochs=5)

# Evaluate the model on the test set
test_steps = len(test_paths) // batch_size
test_gen = data_generator(test_paths, test_labels, batch_size)

test_loss, test_accuracy, test_precision, test_recall, test_f1_score = model.evaluate(test_gen, steps=test_steps)

print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1 Score: {test_f1_score}")

# Save the trained model
model.save("lstm_deepfake_detector.h5")

# Evaluate the model on the test set and get the predicted probabilities
y_pred_prob = model.predict(test_gen, steps=test_steps)

# Compute the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(test_labels[:len(y_pred_prob)], y_pred_prob)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

# Save the plot to a PNG file
plt.savefig('roc_auc_curve.png')




