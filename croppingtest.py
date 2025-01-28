import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import face_recognition
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.reference_embedding = None

    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding for comparison."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        return encodings[0] if encodings else None

    def set_reference_face(self, image: np.ndarray) -> bool:
        """Set reference face for multiple face comparison."""
        embedding = self.get_face_embedding(image)
        if embedding is not None:
            self.reference_embedding = embedding
            return True
        return False

    def get_best_face_match(self, image: np.ndarray, face_locations: List) -> Optional[Tuple]:
        """Find the face that best matches the reference face."""
        if not face_locations or self.reference_embedding is None:
            return face_locations[0] if face_locations else None

        best_match = None
        best_similarity = -1

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for face_location in face_locations:
            encoding = face_recognition.face_encodings(rgb_image, [face_location])[0]
            similarity = cosine_similarity(
                self.reference_embedding.reshape(1, -1),
                encoding.reshape(1, -1)
            )[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face_location

        return best_match if best_similarity > 0.6 else None
class ImageQualityChecker:
    @staticmethod
    def check_blur(image: np.ndarray, threshold: float = 50.0) -> bool:
        """Check if image is blurry."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold

    @staticmethod
    def check_contrast(image: np.ndarray, threshold: float = 30.0) -> bool:
        """Check if image has good contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray) > threshold

    @staticmethod
    def check_resolution(image: np.ndarray, min_resolution: Tuple[int, int] = (256, 256)) -> bool:
        """Check if image meets minimum resolution requirements."""
        height, width = image.shape[:2]
        return width >= min_resolution[0] and height >= min_resolution[1]

class BodyDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def is_full_body(self, image: np.ndarray) -> bool:
        """Detect if image contains full body."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return False

        landmarks = results.pose_landmarks.landmark
        visible_points = [landmark.visibility > 0.5 for landmark in landmarks]
        
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        
        return all(visible_points[landmark.value] for landmark in required_landmarks)

def main():
    root = tk.Tk()
    root.withdraw()
    
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    if not input_folder or not output_folder:
        print("Folders not selected. Exiting.")
        return

    processor = ImageProcessor(input_folder, output_folder)
    processor.process_images()
    
    print("Processing completed. Check the output folder.")

if __name__ == "__main__":
    main()