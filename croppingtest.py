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

class ImageCropper:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.quality_checker = ImageQualityChecker()
        self.body_detector = BodyDetector()

    def calculate_crop_dimensions(
        self, 
        image: np.ndarray, 
        face_location: Tuple, 
        category: int,
        orientation: str
    ) -> Tuple[int, int, int, int]:
        """Calculate crop dimensions based on category and face orientation."""
        height, width = image.shape[:2]
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2

        if category == 2:  # Neck-to-head shots
            # More conservative margins for neck-to-head
            top_margin = int(face_height * 1.0)  # Space above head
            side_margin = int(face_width * 0.8)  # Space on sides
            bottom_margin = int(face_height * 0.5)  # Space below chin
        
        elif category == 3:  # Chest/shoulders-up
            # Larger margins for chest/shoulders
            top_margin = int(face_height * 0.8)  # Less space above head
            side_margin = int(face_width * 1.0)
            bottom_margin = int(face_height * 1.8)  # More space for chest/shoulders

        # Initial crop coordinates
        crop_top = max(0, top - top_margin)
        crop_bottom = min(height, bottom + bottom_margin)
        crop_left = max(0, face_center_x - side_margin)
        crop_right = min(width, face_center_x + side_margin)

        # Ensure square aspect ratio while maintaining face position
        crop_size = max(crop_right - crop_left, crop_bottom - crop_top)
        
        # Adjust crop window to maintain face position
        if category == 2:
            # For neck-to-head, position crop higher
            center_y = top + face_height // 3
        else:
            # For chest/shoulders, position crop lower
            center_y = top + face_height // 2

        center_x = face_center_x

        # Calculate final crop coordinates
        half_size = crop_size // 2
        crop_left = max(0, center_x - half_size)
        crop_right = min(width, center_x + half_size)
        crop_top = max(0, center_y - half_size)
        crop_bottom = min(height, center_y + half_size)

        # Adjust if crop touches boundaries
        if crop_left == 0:
            crop_right = crop_size
        elif crop_right == width:
            crop_left = width - crop_size

        if crop_top == 0:
            crop_bottom = crop_size
        elif crop_bottom == height:
            crop_top = height - crop_size

        return int(crop_top), int(crop_right), int(crop_bottom), int(crop_left)


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