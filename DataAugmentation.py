

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import threading
import os
import time
import random


class DataAugmentation():
    """
    Data Augmentation Functions
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
        Rotate Image
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image, minPercentage = 50, maxPercentage = 75):
        """
        Crop Image To Smaller Size
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_percentage = np.random.randint(minPercentage, maxPercentage) * 0.01
        random_region = (
            int(image_width - crop_win_percentage * image_width) >> 1,
            int(image_height - crop_win_percentage * image_height) >> 1,
            int(image_width + crop_win_percentage * image_width) >> 1,
            int(image_height + crop_win_percentage * image_height) >> 1
        )
        # print(crop_win_percentage)
        return image.crop(random_region)

    @staticmethod
    def randomColor(image):
        """
        Convert Image To Another Color
        @issue: This Methos Mix All Color Change Function, Should Seperate?
        """
        # change saturation
        random_factor = np.random.randint(0, 31) * 0.1
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        # change brightness
        random_factor = np.random.randint(10, 21) * 0.1
        brightness_image = ImageEnhance.Brightness(
            color_image).enhance(random_factor)
        # change contrast
        random_factor = np.random.randint(10, 21) * 0.1
        contrast_image = ImageEnhance.Contrast(
            brightness_image).enhance(random_factor)
        # change Sharpness
        random_factor = np.random.randint(0, 31) * 0.1
        sharpness_image = ImageEnhance.Sharpness(
            contrast_image).enhance(random_factor)
        return sharpness_image

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
        random Gaussian Noise
        """
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            Gaussian Funcition Inner
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.asarray(image)
        img.flags.writeable = True
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def showImage(image):
        image.show()

    @staticmethod
    def saveImage(image, path):
        image.Save(path)

    @staticmethod
    def randomMethod(image, minPercentage = 50, maxPercentage = 75, mean = 0.2, sigma = 0.3):
        rNum = np.random.randint(0,4)
        # print(rNum)
        if rNum == 0:
            return image
        elif rNum == 1:
            return DataAugmentation.randomRotation(image)
        elif rNum == 2:
            return DataAugmentation.randomCrop(image, minPercentage, maxPercentage)
        elif rNum == 3:
            return DataAugmentation.randomColor(image)
        elif rNum == 4:
            return DataAugmentation.randomGaussian(image, mean, sigma)
        else:
            return image



# TempDA = DataAugmentation()
# TempImg = TempDA.openImage("../ai_challenger_scene_train_20170904/scene_train_images_20170904/00000ae5e7fcc87222f1fb6f221e7501558e5b08.jpg")
# TempDA.showImage(TempDA.randomGaussian(TempImg))
