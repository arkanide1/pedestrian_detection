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

def evaluate_model(test_pos, test_neg, model):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # Test positive images
    for filename in os.listdir(test_pos):
        img_path = os.path.join(test_pos, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, WIN_SIZE)
        feature = extract_hog_features(img)
        _, pred = model.predict(np.array([feature]))
        tp += 1 if pred[0][0] == 1 else 0
        fn += 1 if pred[0][0] != 1 else 0
    
    # Test negative images
    for filename in os.listdir(test_neg):
        img = cv2.imread(os.path.join(test_neg, filename))
        if img is not None:
            img = cv2.resize(img, WIN_SIZE)
            feat = extract_hog_features(img)
            _, pred = model.predict(np.array([feat]))
            tn += 1 if pred[0][0] == 0 else 0
            fp += 1 if pred[0][0] != 0 else 0
            
    return tp, fp, tn, fn



def test_hog(test_pos, test_neg):
    # Load HOG model
    svm = cv2.ml.SVM_load("hog_svm.xml")

    # Evaluate HOG model
    tp, fp, tn, fn = evaluate_model(
        test_pos, test_neg, svm)

    # Print HOG results
    print(f"HOG Model Performance:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2%}")
    print("-" * 40)



test_pos = "INRIA_DATASET/test/pos"
test_neg = "INRIA_DATASET/test/neg"
test_hog(test_pos, test_neg)
