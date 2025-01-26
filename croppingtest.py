import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from Katna.image import Image
from Katna.writer import ImageCropDiskWriter
import face_recognition

def remove_white_borders(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Lower the threshold to catch more off-white pixels
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find non-zero points (non-white pixels)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return image
    
    # Get the bounding rectangle of non-white pixels
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add a small margin to ensure we don't cut too tight
    margin = 2
    y = max(0, y - margin)
    h = min(image.shape[0] - y, h + 2*margin)
    
    # Crop the image to remove white borders
    return image[y:y+h, x:x+w]

def crop_with_face_priority(file_path, crop_width, crop_height, output_folder):
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        return False
    
    # Remove white borders
    image = remove_white_borders(image)
    
    # Convert BGR to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image)
    
    if face_locations:
        # Get the first face location
        top, right, bottom, left = face_locations[0]
        
        # Calculate face center
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2
        
        # Calculate face size and add padding for tighter crop
        face_width = right - left
        face_height = bottom - top
        
        # Make crop size relative to face size for tighter framing
        crop_width = int(face_width * 2.5)  # Adjust multiplier for tighter/looser crop
        crop_height = int(face_height * 2.5)  # Adjust multiplier for tighter/looser crop
        
        # Ensure crop dimensions don't exceed image size
        height, width = image.shape[:2]
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)
        
        # Calculate crop coordinates ensuring face is centered
        start_x = max(0, min(face_center_x - crop_width // 2, width - crop_width))
        start_y = max(0, min(face_center_y - crop_height // 2, height - crop_height))
        
        # Perform the crop
        cropped = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        
        # Ensure the cropped image is square
        square_size = min(cropped.shape[0], cropped.shape[1])
        height, width = cropped.shape[:2]
        start_x = (width - square_size) // 2
        start_y = (height - square_size) // 2
        cropped = cropped[start_y:start_y + square_size, start_x:start_x + square_size]
        
        # Save the cropped image
        output_path = os.path.join(output_folder, f"cropped_{os.path.basename(file_path)}")
        cv2.imwrite(output_path, cropped)
        return True
    
    return False

def smart_crop_without_face(image, target_size):
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2
    
    # Calculate crop dimensions
    crop_size = min(width, height)
    crop_size = int(crop_size * 0.8)  # Make crop tighter
    
    # Calculate crop coordinates
    start_x = center_x - (crop_size // 2)
    start_y = center_y - (crop_size // 2)
    
    # Ensure coordinates are within bounds
    start_x = max(0, min(start_x, width - crop_size))
    start_y = max(0, min(start_y, height - crop_size))
    
    # Perform the crop
    cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    return cropped

def main():
    # Open file dialog to select input and output directories
    root = tk.Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    output_folder = filedialog.askdirectory(title="Select Output Folder")

    if not input_folder or not output_folder:
        print("Folders not selected. Exiting.")
        return

    # Initialize Katna Image module
    img_module = Image()
    diskwriter = ImageCropDiskWriter(location=output_folder)

    # Process each image
    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_file_path = os.path.join(input_folder, file_name)
        print(f"Processing: {file_name}")
        
        # Read the image
        image = cv2.imread(input_file_path)
        if image is None:
            print(f"Skipping file: {file_name} (not a valid image)")
            continue

        # Remove white borders
        image = remove_white_borders(image)
        
        # Get dimensions after removing borders
        height, width = image.shape[:2]
        square_size = min(width, height)

        # Try face-priority cropping first
        if not crop_with_face_priority(input_file_path, square_size, square_size, output_folder):
            try:
                # If no face detected, use smart cropping
                cropped = smart_crop_without_face(image, square_size)
                
                # Save the cropped image
                output_path = os.path.join(output_folder, f"cropped_{file_name}")
                cv2.imwrite(output_path, cropped)
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                
                # Fallback to Katna if smart crop fails
                try:
                    temp_path = os.path.join(output_folder, "temp_" + file_name)
                    cv2.imwrite(temp_path, image)
                    
                    img_module.crop_image(
                        file_path=temp_path,
                        crop_width=square_size,
                        crop_height=square_size,
                        num_of_crops=1,
                        writer=diskwriter,
                        filters=None,
                        down_sample_factor=1
                    )
                    
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Katna fallback failed for {file_name}: {str(e)}")

    print("Cropping completed. Check the output folder.")

if __name__ == "__main__":
    main()