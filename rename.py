import os
from tkinter import Tk, filedialog

def rename_images(folder_path, base_name):
    """
    Renames all images in the specified folder to the format base_name+number and moves them to a new folder.

    :param folder_path: Path to the folder containing images.
    :param base_name: Base name for the images.
    """
    try:
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

        # Create a new folder with the base name
        new_folder_path = os.path.join(folder_path, base_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Get list of files in the folder
        files = os.listdir(folder_path)

        # Filter images based on extensions
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

        # Rename and move images
        for index, image in enumerate(images, start=1):
            old_path = os.path.join(folder_path, image)
            new_name = f"{base_name}{index}{os.path.splitext(image)[1]}"
            new_path = os.path.join(new_folder_path, new_name)
            os.rename(old_path, new_path)

        print(f"Renamed and moved {len(images)} images successfully to {new_folder_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    Tk().withdraw()  # Hide the main tkinter window
    folder = filedialog.askdirectory(title="Select the folder containing images")
    if not folder:
        print("No folder selected. Exiting...")
    else:
        name = input("Enter the base name for the images: ")
        rename_images(folder, name)
