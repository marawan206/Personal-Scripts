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

if __name__ == "__main__":
    main()