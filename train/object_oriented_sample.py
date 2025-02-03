import os
import cv2
from PIL import Image

class ImageProcessor:
    def _init_(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def get_image_paths(self):
        """Generator to yield paths of PNG images in the input folder."""
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.png'):
                yield os.path.join(self.input_folder, filename)

    def process_image(self, image_path):
        """Process an image and return a binary mask."""
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, 1, 1, cv2.THRESH_BINARY)
        return mask

    def save_image(self, image, filename):
        """Save the processed image to the output folder."""
        output_path = os.path.join(self.output_folder, filename)
        Image.fromarray(image).save(output_path)

    def process_all(self):
        """Process all images in the input folder."""
        for image_path in self.get_image_paths():
            filename = os.path.basename(image_path)
            processed_image = self.process_image(image_path)
            self.save_image(processed_image, filename)

class GrayscaleProcessor(ImageProcessor):
    """Processor to save images as grayscale versions."""
    def process_image(self, image_path):
        """Process an image and return it as grayscale."""
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

# Example usage
input_folder = 'C:/remote_sensing/data/1_data/groundtruth/original/grayscale'
output_folder = 'C:/remote_sensing/data/1_data/groundtruth/processed/grayscale'
processor = GrayscaleProcessor(input_folder, output_folder)
processor.process_all()