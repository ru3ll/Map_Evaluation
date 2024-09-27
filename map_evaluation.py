#!/usr/bin/python3

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img, (120, 64))
    return img

def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def align_images(image1, image2):
    # Find keypoints and descriptors
    kp1, des1 = extract_features(image1)
    kp2, des2 = extract_features(image2)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))

    # Check if there are enough matches
    if len(matches) < 4:
        print("Not enough matches found to compute homography.")
        return image1  # Return the original image if not enough matches

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Align image1 to image2
    height, width = image2.shape
    aligned_image1 = cv2.warpPerspective(image1, M, (width, height))

    return aligned_image1

def quality_metrics(aligned_image, reference_image):
    ssim_value = ssim(aligned_image, reference_image)
    mse_value = np.mean((aligned_image - reference_image) ** 2)
    return ssim_value, mse_value

def main(generated_map_path, standard_map_path, ssim_threshold=0.8, mse_threshold=1000):
    generated_map = preprocess_image(generated_map_path)
    standard_map = preprocess_image(standard_map_path)

    aligned_map = align_images(generated_map, standard_map)

    ssim_value, mse_value = quality_metrics(aligned_map, standard_map)

    print(f"SSIM: {ssim_value}, MSE: {mse_value}")
    
    if ssim_value > ssim_threshold and mse_value < mse_threshold:
        print("Map quality acceptable")
    else:
        print("Map quality unacceptable")

if __name__ == "__main__":
    main('maps/sim2_save.pgm', 'maps/sim1_save.pgm')
