# Emotion-Detection-with-OpenCV-and-Tensorflow
A basic tensorflow model fused with OpenCV Haarcascade to detect emotions in live video or images.

# Dataset
The model was trained on the FER-2013 Dataset which is avaliable on Kaggle: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

# Model
The model architecture is a simple CNN with 2 Conv2D layers followed by a flatten and 2 Dense layers. The last Dense layer has 7 neurons matching the number of classes in the dataset.
