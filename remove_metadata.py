import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select folder with images")
    return folder_path

def remove_metadata(input_folder):
    if not input_folder:  # If no folder was selected
        print("No folder selected. Exiting...")
        return 0
    
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    
    # Counter for processed files
    processed = 0
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            try:
                # Open the image
                input_path = os.path.join(input_folder, filename)
                image = Image.open(input_path)
                
                # Create a new image without metadata
                data = list(image.getdata())
                image_without_exif = Image.new(image.mode, image.size)
                image_without_exif.putdata(data)
                
                # Save the new image (overwriting the original)
                image_without_exif.save(input_path, quality=95)
                
                processed += 1
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return processed

if __name__ == "__main__":
    print("Select the folder containing images to remove metadata...")
    input_folder = select_folder()
    
    try:
        processed_files = remove_metadata(input_folder)
        print(f"\nCompleted! Processed {processed_files} files.")
        if processed_files > 0:
            print(f"All images in {input_folder} have been cleaned of metadata.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    input("\nPress Enter to exit...")