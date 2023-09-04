import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import cv2
import argparse
import glob
from tqdm import tqdm

class CropFaces():
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.get_images()
        self.init_detector()

    def get_images(self):
        input_images = glob.glob(self.input_dir + '*.jpg')
        print(f'Total number of input images: {len(input_images)}')
        self.input_images = input_images
    
    def init_detector(self):
        self.detector = MTCNN()
        print('Model is initialized.')
    
    def detect_faces(self):
        for img in self.input_images:
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            detected = self.detector.detect_faces(image)
            if len(detected) > 1:
                print(detected)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir')
parser.add_argument('-o', '--output_dir')
args = parser.parse_args()
assert os.path.exists(args.input_dir), 'Input path does not exists.'
if os.path.exists(args.output_dir) == False:
    os.makedirs(args.output_dir)
    print('Output dir did not exists, so created.')

class_obj = CropFaces(args.input_dir, args.output_dir)
class_obj.detect_faces()


# Demo string
# python crop_faces.py -i '../../face_files/part1/' -o '../../face_files/part_1_cropped/'