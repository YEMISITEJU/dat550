# dat550
How to reproduce the results

## CNN

* Download 10.zip file from https://www.kaggle.com/competitions/deepfake-detection-challenge/data
* Run jupyter notebook file dfdc_li.ipynb which contains all the code needed to reproduce the result.
* Configure train_dir, test_dir etc according to your folder names. 
* uncomment the following lines in dfdc_li.ipynb when preprocessing videos to extract faces from videos
#extract_jpg_frames_from_train_videos(train_dir, metadata_file,fake_image_folder, real_image_folder, frames_per_video)
#extract_jpg_frames_from_test_videos(test_dir, test_image_folder, frames_per_video)

## LSTM

Download the dataset (https://ai.facebook.com/datasets/dfdc/)

Run pre_processing_yy.py (edit the path to file directory)

Run train_lstm_model.py (edit the path to file directory)

### CNN Siamese
The code is in two notebooks ; pre_processing_hc.ipynb and siames_network_hc.ipynb (Latest version is in Github, had to change the classification class for easy reproduction last minute) 
* Download dataset from kaggle: https://www.kaggle.com/competitions/deepfake-detection-challenge/data I used the sample data, download button is on the bottom right
* Run the preprocessing notebook to extract faces from the dataset
* Run the siames_network_hc to train, evaluate and predict
