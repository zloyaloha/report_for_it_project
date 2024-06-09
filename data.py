import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
import string
from PIL import Image

import numpy as np

def resize_images(image_path, target_size):
    image = cv2.imread(image_path)
    print(image_path)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_image

def normalize_image(image_path, method='min_max'):
    image = cv2.imread(image_path)
    if method == 'min_max':
        a = np.min(image)
        b = np.max(image)
        normalized_image = (image - a) / (b - a)
    elif method == 'standard':
        mean = np.mean(image)
        std = np.std(image)
        normalized_image = (image - mean) / std
    else:
        raise ValueError("Недопустимый метод нормализации. Используйте 'min_max' или 'standard'.")
    
    return normalized_image

def generate_alphanum_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    rand_string = ''.join(random.sample(letters_and_digits, length))
    return rand_string

def rotate_image_and_labels(image_path, label_path, image_dir, label_dir):

    image = cv2.imread(image_path)

    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f]

    height, width, _ = image.shape

    M = np.float32([[-1, 0, width], [0, -1, height]])

    rotated_image = cv2.warpAffine(image, M, (width, height))

    rotated_labels = []
    for label in labels:
        class_id = label[0]
        rotated_points = []
        rotated_points.append(class_id)
        for x, y in zip(label[1::2], label[2::2]):
            new_x = 1 - float(x)
            new_y = 1 - float(y)
            rotated_points.extend([str(new_x), str(new_y)])
        rotated_labels.append(rotated_points)

    rand_string = generate_alphanum_random_string(8)
    new_image_path = image_dir + rand_string + ".jpg"
    cv2.imwrite(new_image_path, rotated_image)
    new_label_path = label_dir + rand_string + ".txt"
    with open(new_label_path, 'w') as f:
        for label in rotated_labels:
            f.write(' '.join(label) + '\n')

    print(f'Saved augmented data: {os.path.basename(new_image_path)}')

def mark_polygon_on_image(image_path, points_file_path):
    image = cv2.imread(image_path)

    with open(points_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        coordinates = line.strip().split()
        class_id = int(coordinates[0])
        points = [[float(x) * image.shape[1], float(y) * image.shape[0]] for x, y in zip(coordinates[1::2], coordinates[2::2])]

        points = np.array(points, dtype=np.int32)

        if class_id == 0:
            color = (0, 255, 0)
        elif class_id == 1:
            color = (255, 0, 0)
        elif class_id == 2:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        cv2.polylines(image, [points], True, color, thickness=2)

    cv2.imwrite('output_image.jpg', image)

images_dir = 'markedData/images/'
label_dir = 'markedData/labels/'

images_dir_rot = 'markedData/imagesRotated/'
label_dir_rot = 'markedData/labelsRotated/'

# # Поворот
# for filename in os.listdir('markedData/images/'):
#     if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
#         image_path = os.path.join(images_dir, filename)
#         label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
#         rotate_image_and_labels(image_path, label_path, images_dir_rot, label_dir_rot)
    
# # Проверка корректности поворота полигона
# mark_polygon_on_image("markedData/imagesRotated/0ANmIO6L.jpg", "markedData/labelsRotated/0ANmIO6L.txt")

# #Нормализация
# for filename in os.listdir('markedData/images/'):
#     if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
#         image_path = os.path.join(images_dir, filename)
#         normalized_image_min_max = normalize_image(image_path, method='min_max')
#         cv2.imwrite("markedData/normalized/" + generate_alphanum_random_string(8) + ".jpg", (normalized_image_min_max * 255).astype(np.uint8))

# # Масштабирование
# target_size = (600, 300)
# for filename in os.listdir('markedData/normalized/'):
#     if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
#         image_path = os.path.join("markedData/normalized/", filename)
#         resized_images = resize_images(image_path, target_size)
#         cv2.imwrite("markedData/resized/" + filename, resized_images)