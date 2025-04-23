import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import imutils

class PedestrianDetector:
    def __init__(self):
        # Initialize HOG parameters
        self.win_size = (64, 128)  # Standard window size for pedestrian detection
        self.block_size = (16, 16)  # 2x2 cells
        self.block_stride = (8, 8)  # 50% overlap
        self.cell_size = (8, 8)     # 8x8 pixels per cell
        self.nbins = 9              # 9 orientation bins
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, 
                                    self.block_stride, self.cell_size, self.nbins)
        self.svm = LinearSVC(C=0.01, max_iter=10000, random_state=42)
        
    def load_dataset(self, pos_path, neg_path):
        """Load positive and negative samples from directories"""
        features = []
        labels = []
        
        # Load positive samples (pedestrians)
        print("[INFO] Loading positive samples...")
        for filename in os.listdir(pos_path):
            img_path = os.path.join(pos_path, filename)
            if not filename.endswith(('.jpg', '.png')):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize if necessary (should already be 64x128 in INRIA)
            img = cv2.resize(img, self.win_size)
            
            # Compute HOG features
            hog_features = self.hog.compute(img)
            features.append(hog_features.flatten())
            labels.append(1)
        
        # Load negative samples (non-pedestrians)
        print("[INFO] Loading negative samples...")
        for filename in os.listdir(neg_path):
            img_path = os.path.join(neg_path, filename)
            if not filename.endswith(('.jpg', '.png')):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize if necessary (should already be 64x128 in INRIA)
            img = cv2.resize(img, self.win_size)
            
            # Compute HOG features
            hog_features = self.hog.compute(img)
            features.append(hog_features.flatten())
            labels.append(-1)
            
        return np.array(features), np.array(labels)
    
    def train(self, X_train, y_train):
        """Train the SVM classifier"""
        print("[INFO] Training SVM...")
        start_time = time.time()
        self.svm.fit(X_train, y_train)
        print(f"[INFO] Training completed in {time.time() - start_time:.2f} seconds")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        print("[INFO] Evaluating...")
        y_pred = self.svm.predict(X_test)
        print(classification_report(y_test, y_pred))
        
    def detect(self, image, threshold=0.5, scale_step=1.05, window_step=8):
        """
        Detect pedestrians in an image using sliding window approach
        Args:
            image: Input image (BGR format)
            threshold: SVM decision threshold (higher = fewer false positives)
            scale_step: Scale factor for image pyramid
            window_step: Step size for sliding window (pixels)
        Returns:
            image_with_boxes: Image with bounding boxes drawn
            rects: List of detected rectangles (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orig = image.copy()
        
        # Initialize variables for multi-scale detection
        rects = []
        confidences = []
        
        # Image pyramid for multi-scale detection
        for scale in np.linspace(0.5, 1.5, 8):
            # Resize the image
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            current_scale = gray.shape[1] / float(resized.shape[1])
            
            # Stop if resized image is smaller than window size
            if resized.shape[0] < self.win_size[1] or resized.shape[1] < self.win_size[0]:
                break
                
            # Sliding window
            for y in range(0, resized.shape[0] - self.win_size[1], window_step):
                for x in range(0, resized.shape[1] - self.win_size[0], window_step):
                    # Extract the window
                    window = resized[y:y + self.win_size[1], x:x + self.win_size[0]]
                    
                    # Compute HOG features
                    hog_features = self.hog.compute(window).flatten()
                    
                    # Predict using SVM
                    confidence = self.svm.decision_function([hog_features])[0]
                    
                    # If detection passes threshold, record it
                    if confidence > threshold:
                        # Scale coordinates back to original image
                        start_x = int(x * current_scale)
                        start_y = int(y * current_scale)
                        end_x = int((x + self.win_size[0]) * current_scale)
                        end_y = int((y + self.win_size[1]) * current_scale)
                        
                        rects.append((start_x, start_y, end_x - start_x, end_y - start_y))
                        confidences.append(confidence)
        
        # Apply non-maxima suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(rects, confidences, threshold, 0.3)
        
        # Draw final bounding boxes
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y, w, h) = rects[i]
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        return orig, rects
    
    def save_model(self, path):
        """Save the trained model"""
        import joblib
        joblib.dump((self.hog, self.svm), path)
        print(f"[INFO] Model saved to {path}")
        
    def load_model(self, path):
        """Load a trained model"""
        import joblib
        self.hog, self.svm = joblib.load(path)
        print(f"[INFO] Model loaded from {path}")

def main():
    # Initialize detector
    detector = PedestrianDetector()
    
    # Paths to your dataset
    train_pos = "INRIA_DATASET/train/pos"
    train_neg = "INRIA_DATASET/train/neg"
    test_pos = "INRIA_DATASET/test/pos"
    test_neg = "INRIA_DATASET/test/neg"
    
    # Load and prepare dataset
    print("[INFO] Loading training data...")
    X_train, y_train = detector.load_dataset(train_pos, train_neg)
    
    print("[INFO] Loading test data...")
    X_test, y_test = detector.load_dataset(test_pos, test_neg)
    
    # Train the model
    detector.train(X_train, y_train)
    
    # Evaluate on test set
    detector.evaluate(X_test, y_test)
    
    # Save the model
    detector.save_model("pedestrian_detector.pkl")
    
    # Test on larger images
    test_image_paths = ["test_images/image1.jpg", "test_images/image2.jpg"]  # Add your test images
    
    for test_path in test_image_paths:
        print(f"[INFO] Processing {test_path}...")
        image = cv2.imread(test_path)
        if image is None:
            continue
            
        # Detect pedestrians
        result, rects = detector.detect(image, threshold=0.5)
        
        # Show results
        cv2.imshow("Detections", result)
        cv2.waitKey(0)
        
        # Save results
        output_path = os.path.join("output", os.path.basename(test_path))
        cv2.imwrite(output_path, result)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()