import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import gc

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select main folder with images")
    return folder_path

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_image(file_info):
    input_path, filename = file_info
    try:
        # Use context manager for better memory management
        with Image.open(input_path) as image:
            # Create a new image without metadata
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)
            image_without_exif.save(input_path, quality=95)
            
            # Force cleanup
            del data
            del image_without_exif
            gc.collect()
            
        return True, input_path
    except Exception as e:
        return False, f"Error processing {filename}: {str(e)}"
        
def main():
    try:
        print("=== Image Metadata Removal Tool ===")
        print("\nSelect the main folder containing images to remove metadata...")
        input_folder = select_folder()
        
        processed_files = remove_metadata(input_folder)
        
        if processed_files > 0:
            print(f"\nAll images in {input_folder} and its subfolders have been cleaned of metadata.")
        
        print("\nPress Enter to exit...")
        input()
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)

if __name__ == "__main__":
    # Set up console encoding for Windows
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    main()