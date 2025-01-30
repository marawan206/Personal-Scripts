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

    def detect_face_orientation(self, image: np.ndarray) -> Tuple[str, float]:
        """Detect if face is front-facing or profile."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return "unknown", 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        # Calculate face orientation using nose and ear landmarks
        return "front", 0.0

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

    def verify_crop_quality(self, cropped: np.ndarray) -> bool:
        """Verify the quality of the cropped image."""
        if cropped is None or cropped.size == 0:
            return False

        height, width = cropped.shape[:2]
        
        # Verify square aspect ratio (allow 2 pixels difference)
        if abs(height - width) > 2:
            return False

        # Verify minimum size
        if height < 256 or width < 256:
            return False

        return True

    def adjust_crop_boundaries(
        self,
        image: np.ndarray,
        top: int,
        right: int,
        bottom: int,
        left: int
    ) -> Tuple[int, int, int, int]:
        """Adjust crop boundaries to ensure they're within image dimensions."""
        height, width = image.shape[:2]
        
        # Ensure boundaries are within image dimensions
        top = max(0, min(top, height - 1))
        bottom = max(0, min(bottom, height))
        left = max(0, min(left, width - 1))
        right = max(0, min(right, width))
        
        # Ensure square aspect ratio
        crop_width = right - left
        crop_height = bottom - top
        
        if crop_width != crop_height:
            size = min(crop_width, crop_height)
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            half_size = size // 2
            left = center_x - half_size
            right = center_x + half_size
            top = center_y - half_size
            bottom = center_y + half_size
            
            # Final boundary check
            if left < 0:
                shift = -left
                left += shift
                right += shift
            elif right > width:
                shift = right - width
                left -= shift
                right -= shift
                
            if top < 0:
                shift = -top
                top += shift
                bottom += shift
            elif bottom > height:
                shift = bottom - height
                top -= shift
                bottom -= shift
        
        return int(top), int(right), int(bottom), int(left)

    def crop_image(
        self, 
        image: np.ndarray, 
        face_location: Tuple, 
        category: int
    ) -> Optional[np.ndarray]:
        """Crop image based on category and face location."""
        try:
            # Get face orientation
            orientation, _ = self.face_analyzer.detect_face_orientation(image)
            
            # Calculate initial crop dimensions
            top, right, bottom, left = self.calculate_crop_dimensions(
                image, face_location, category, orientation
            )
            
            # Adjust boundaries to ensure they're valid
            top, right, bottom, left = self.adjust_crop_boundaries(
                image, top, right, bottom, left
            )
            
            # Perform the crop
            cropped = image[top:bottom, left:right]
            
            # Verify the crop quality
            if self.verify_crop_quality(cropped):
                # Ensure exact square aspect ratio
                height, width = cropped.shape[:2]
                if height != width:
                    size = min(height, width)
                    start_y = (height - size) // 2
                    start_x = (width - size) // 2
                    cropped = cropped[start_y:start_y+size, start_x:start_x+size]
                
                return cropped
            
            return None
            
        except Exception as e:
            logger.error(f"Error during cropping: {str(e)}")
            return None

    def remove_white_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove white borders from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        
        if coords is None:
            return image
            
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]

class ImageProcessor:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cropper = ImageCropper()
        self.face_analyzer = FaceAnalyzer()
        self.body_detector = BodyDetector()
        self.processed_images = set()
        
        # Track counts for each category
        self.category_counts = {
            'C1': 0,
            'C2': 0,
            'C3': 0
        }
        
        os.makedirs(output_folder, exist_ok=True)

    def determine_category(self, total_images: int, current_image: str) -> int:
        """
        Determine which category (2 or 3) the image should be assigned to.
        Aims for roughly 60% C3 (chest/shoulders) and 40% C2 (neck-to-head).
        """
        remaining_images = total_images - len(self.processed_images)
        target_c2 = int(total_images * 0.4)  # 40% for C2
        target_c3 = int(total_images * 0.6)  # 60% for C3

        # If we haven't met C2 target and we're not at risk of having too few C3s
        if (self.category_counts['C2'] < target_c2 and 
            remaining_images > (target_c3 - self.category_counts['C3'])):
            return 2
        # If we haven't met C3 target
        elif self.category_counts['C3'] < target_c3:
            return 3
        # Default to C2 if we somehow haven't assigned yet
        else:
            return 2

    def process_images(self):
        """Process all images in the input folder."""
        image_files = [f for f in os.listdir(self.input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            logger.error("No images found in input folder")
            return

        # Sort files to ensure consistent processing
        image_files.sort()
        total_images = len(image_files)

        # First pass: analyze images and set reference face
        full_body_images = []
        reference_face_set = False
        
        for filename in image_files:
            filepath = os.path.join(self.input_folder, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                continue

            if not reference_face_set:
                if self.face_analyzer.set_reference_face(image):
                    reference_face_set = True

            if self.body_detector.is_full_body(image):
                full_body_images.append(filename)
                logger.info(f"Full body detected in: {filename}")

        # Second pass: process and categorize images
        for filename in image_files:
            if filename in self.processed_images:
                continue

            filepath = os.path.join(self.input_folder, filename)
            image = cv2.imread(filepath)
            
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                continue

            face_location = self.face_analyzer.get_best_face_match(image, face_locations)
            
            if not face_location:
                continue

            # Handle C1 (wide shots) first
            if filename in full_body_images and self.category_counts['C1'] < 2:
                output_filename = f"C1_{filename}"
                cv2.imwrite(os.path.join(self.output_folder, output_filename), image)
                self.category_counts['C1'] += 1
                self.processed_images.add(filename)
                logger.info(f"Saved wide shot: {output_filename}")
                continue

            # Determine category (2 or 3) based on distribution targets
            category = self.determine_category(total_images, filename)
            
            cropped = self.cropper.crop_image(image, face_location, category)
            if cropped is not None:
                output_filename = f"C{category}_{filename}"
                cv2.imwrite(os.path.join(self.output_folder, output_filename), cropped)
                self.category_counts[f'C{category}'] += 1
                self.processed_images.add(filename)
                logger.info(f"Saved category {category}: {output_filename}")

        # Log final distribution
        logger.info(f"Final distribution: C1: {self.category_counts['C1']}, "
                   f"C2: {self.category_counts['C2']}, "
                   f"C3: {self.category_counts['C3']}")
"""
C1: Wide shots (full body)
C2: Neck-to-head shots
C3: Chest/shoulders-up shots
"""
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