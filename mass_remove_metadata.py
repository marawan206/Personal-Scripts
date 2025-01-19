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

def remove_metadata(input_folder):
    if not input_folder:  # If no folder was selected
        print("No folder selected. Exiting...")
        return 0
    
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    
    print("\nCollecting files...")
    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(supported_formats):
                input_path = os.path.join(root, filename)
                image_files.append((input_path, filename))
    
    if not image_files:
        print("No supported image files found.")
        return 0
    
    total_files = len(image_files)
    print(f"\nFound {total_files} images to process")
    
    # Use multiprocessing to process images
    num_processes = cpu_count()  # Get number of CPU cores
    print(f"Using {num_processes} CPU cores for processing...\n")
    
    # Process in batches
    batch_size = 1000
    processed_count = 0
    errors = []
    
    with Pool(processes=num_processes) as pool:
        # Process each batch with progress bar
        for batch in chunks(image_files, batch_size):
            results = list(tqdm(
                pool.imap(process_image, batch),
                total=len(batch),
                desc="Processing images",
                unit="image"
            ))
            
            # Count successes and collect errors
            for success, message in results:
                if success:
                    processed_count += 1
                else:
                    errors.append(message)
    
    # Report results
    print("\nProcessing completed!")
    print(f"Successfully processed: {processed_count} files")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
    
    return processed_count

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