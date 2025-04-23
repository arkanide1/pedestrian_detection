import cv2
import numpy as np
import os

WIN_SIZE = (64, 128)  # Standard window size for human detection
BLOCK_SIZE = (16, 16)  # Size of blocks for feature analysis
BLOCK_STRIDE = (8, 8)  # How much blocks move across the image (over)
CELL_SIZE = (8, 8)     # Size of cells that make up blocks
HISTOGRAM_BINS = 9     # Number of gradient directions to track


def extract_hog_features(img):
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image
    These features help the computer understand shapes and patterns
    """
    # Create HOG descriptor with our configuration
    hog = cv2.HOGDescriptor(
        WIN_SIZE,
        BLOCK_SIZE,
        BLOCK_STRIDE,
        CELL_SIZE,
        HISTOGRAM_BINS
    )
    
    # Calculate features and flatten to 1D array for the SVM
    return hog.compute(img).flatten()


def load_dataset(pos_folder, neg_folder):
    """
    Load positive (human) and negative (non-human) images,
    extract features, and create labels for training
    """
    features = []
    labels = []
    
    for filename in os.listdir(pos_folder):
        img_path = os.path.join(pos_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, WIN_SIZE)
        if img is not None:
            features.append(extract_hog_features(img))
            labels.append(1)  # Positive samples labeled as 1

    for filename in os.listdir(neg_folder):
        img_path = os.path.join(neg_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, WIN_SIZE)
        if img is not None:
            features.append(extract_hog_features(img))
            labels.append(0)  # Negative samples labeled as 0

    return np.array(features, dtype=np.float32), np.array(labels)


def train_hog_model():
    """
    Main training function that loads data and trains the SVM classifier
    """
    # Dataset paths (update these if your folder structure is different)
    train_pos = "INRIA_DATASET/train/pos"
    train_neg = "INRIA_DATASET/train/neg"

    # Load training data
    print("Loading training images...")
    X_train, y_train = load_dataset(train_pos, train_neg)

    # Set up Support Vector Machine (SVM) classifier
    print("Setting up SVM classifier...")
    svm = cv2.ml.SVM_create() 
    svm.setType(cv2.ml.SVM_C_SVC)    # Classification type
    svm.setKernel(cv2.ml.SVM_LINEAR) # Linear kernel works well with HOG

    # Train the model
    print("Training model...")
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)  # cv2.ml.ROW_SAMPLE each row corresponds to a feature vector of one image

    # Save the trained model
    svm.save("hog_svm.xml")
    print("Training complete! Model saved as hog_svm.xml")



train_hog_model()