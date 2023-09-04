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
import pickle

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

    def crop_and_save(self, image, co_ords, image_name):
        im_crop = image.crop((co_ords[0], co_ords[1], co_ords[0] + co_ords[2], co_ords[1] + co_ords[3]))
        # image = image[co_ords[0]:(co_ords[0] + co_ords[2]), co_ords[1]:(co_ords[1] + co_ords[3])]
        # cv2.imwrite(self.output_dir+image_name, im_crop)
        im_crop.save(self.output_dir+image_name)

    
    def detect_faces(self):
        issues = []
        for img in self.input_images:
            img_name = img.split('/')[-1]
            image = Image.open(img)
            img_arr = np.array(image)
            try:
                detected = self.detector.detect_faces(img_arr)
                if len(detected) > 1:
                    for detections in range(len(detected)):
                        data = detected[detections]['box']
                        image_name_ = img_name.split('.')[0] + str(detections)
                        self.crop_and_save(image, data, image_name_ + '_' + '.jpg')    
                elif len(detected) == 1:
                    data = detected[0]['box']
                    self.crop_and_save(image, data, img_name)
            except:
                print(img)
                issues.append(img)
        with open('./issues.pkl', 'wb') as handle:
            pickle.dump(issues, handle)

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
# python crop_faces.py -i ../../face_files/wild_faces/part1/ -o ../../face_files/part_1_cropped/