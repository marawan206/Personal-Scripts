import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import sys
import tkinter as tk
from tkinter import filedialog

def load_and_preprocess_image(img_path, target_size=(299, 299)):
    try:
        img = keras_image.load_img(img_path, target_size=target_size)
        img_data = keras_image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def extract_features(directory, model):
    features = {}
    filenames = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            filepath = os.path.join(directory, filename)
            img_data = load_and_preprocess_image(filepath)
            if img_data is not None:
                preds = model.predict(img_data)
                features[filename] = preds.flatten()
                filenames.append(filename)
            else:
                print(f"Skipping image {filename}")
    return features, filenames

def find_similar_images(features, filenames, similarity_threshold=0.89):
    # Compute the cosine similarity matrix
    feature_vectors = np.array(list(features.values()))
    similarity_matrix = cosine_similarity(feature_vectors)
    # Set diagonal to zero to ignore self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    
    # Find pairs of images that are above the similarity threshold
    similar_images = {}
    visited = set()
    for i in range(len(filenames)):
        if filenames[i] in visited:
            continue
        similars = []
        for j in range(len(filenames)):
            if i != j and similarity_matrix[i][j] > similarity_threshold:
                similars.append(filenames[j])
                visited.add(filenames[j])
        if similars:
            similars.append(filenames[i])
            similar_images[filenames[i]] = list(set(similars))
            visited.add(filenames[i])
    return similar_images

def remove_similar_images(similar_images, directory):
    images_to_remove = set()
    for key, similars in similar_images.items():
        # Keep one image, remove the rest
        similars.remove(key)
        for img in similars:
            images_to_remove.add(img)
    
    for img in images_to_remove:
        filepath = os.path.join(directory, img)
        try:
            os.remove(filepath)
            print(f"Removed {filepath}")
        except Exception as e:
            print(f"Could not remove {filepath}: {e}")

def main():
    # Use tkinter to open a directory selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the directory selection dialog
    directory = filedialog.askdirectory(title="Select Directory Containing Images")

    # Check if a directory was selected
    if not directory:
        print("No directory selected. Exiting.")
        sys.exit(1)

    # Set the similarity threshold (adjust if necessary)
    similarity_threshold = 0.89  # Between 0 and 1

    # Load pre-trained model (InceptionV3) and remove the classification layers
    print("Loading pre-trained model...")
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print("Model loaded.")

    # Extract features from images
    print("Extracting features from images...")
    features, filenames = extract_features(directory, model)
    print("Feature extraction completed.")

    # Find similar images
    print("Finding similar images...")
    similar_images = find_similar_images(features, filenames, similarity_threshold)
    print(f"{len(similar_images)} groups of similar images found.")

    # Remove similar images
    print("Removing similar images...")
    remove_similar_images(similar_images, directory)
    print("Similar images have been removed.")

if __name__ == "__main__":
    main()