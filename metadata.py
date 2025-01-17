import json
import os
from tkinter import Tk, filedialog, Button, Label
from PIL import Image

def get_png_metadata(file_path):
    """
    Extracts metadata from a PNG file.

    :param file_path: Path to the PNG file
    :return: Dictionary containing metadata and basic image info
    """
    try:
        with Image.open(file_path) as img:
            metadata = {
                "File Name": os.path.basename(file_path),
                "File Path": file_path,
                "Format": img.format,
                "Mode": img.mode,
                "Size": img.size,
                "Metadata": img.info if img.info else "No metadata found"
            }
            return metadata
    except Exception as e:
        return {"Error": str(e)}

def save_metadata(metadata, output_dir):
    """
    Saves the metadata to a JSON file.

    :param metadata: Dictionary containing metadata
    :param output_dir: Directory to save the output files
    """
    file_name = os.path.splitext(metadata["File Name"])[0]
    
    # Save as a JSON file
    json_file_path = os.path.join(output_dir, f"{file_name}_metadata.json")
    with open(json_file_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    print(f"Metadata saved as: {json_file_path}")

def comfyui_quick_data_extract():
    """
    Handles the ComfyUI Quick Data extract option.
    """
    input_dir = r""
    output_dir = r""

    file_path = filedialog.askopenfilename(
        title="Select a PNG file from ComfyUI Output",
        initialdir=input_dir,
        filetypes=[("PNG Files", "*.png")]
    )
    if not file_path:
        print("No file selected.")
        return

    metadata = get_png_metadata(file_path)
    save_metadata(metadata, output_dir)

def choose_another():
    """
    Handles the Choose Another option.
    """
    # Open file dialog to select a PNG file
    file_path = filedialog.askopenfilename(
        title="Select a PNG file",
        filetypes=[("PNG Files", "*.png")]
    )
    if not file_path:
        print("No file selected.")
        return

    # Choose output directory
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return

    metadata = get_png_metadata(file_path)
    save_metadata(metadata, output_dir)

def main():
    # Create the main window
    root = Tk()
    root.title("PNG Metadata Extractor")
    root.geometry("400x200")

    # Add a label
    label = Label(root, text="Choose an option to extract metadata", font=("Arial", 14))
    label.pack(pady=20)

    # Add buttons
    button1 = Button(
        root, 
        text="Quick Data Extract", 
        command=comfyui_quick_data_extract, 
        width=30, 
        height=2, 
        bg="lightblue"
    )
    button1.pack(pady=10)

    button2 = Button(
        root, 
        text="Choose Another", 
        command=choose_another, 
        width=30, 
        height=2, 
        bg="lightgreen"
    )
    button2.pack(pady=10)

    # Run the GUI loop
    root.mainloop()

if __name__ == "__main__":
    main()
