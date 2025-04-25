import argparse
import os
import cv2
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


class VerifyModel():

    def __init__(self):
        self.model_path = ""
        self.input_folder = ""
        self.output_folder = ""
        self.task = "detect"
        self.version = "v1.0"

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description = "Test YOLO Model on an Image.")
        parser.add_argument("--model_path", type=str, help="Path to the exported model.")
        parser.add_argument("--input_folder", type=str, help="Path to the input folder containing images.")
        parser.add_argument("--output_folder", type=str, help="Path to the folder to save bounding images.")
        parser.add_argument("--task", type=str, default="detect", help="Kind of task to be performed. Default: detect.")
        parser.add_argument("--version", default="v1.0", type=str, help="The version of the expored model.")
        args = parser.parse_args()

        return args

    def start(self, args):

        if args.version:
            self.version = args.version
        self.model_path = args.model_path
        self.input_folder = args.input_folder
        self.output_folder = args.output_folder
        self.task = args.task
        
        # Load model
        model = YOLO(self.model_path, task=self.task)

        # Make sure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Run inference on all images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.jfif')
        image_files = [f for f in Path(self.input_folder).iterdir() if f.suffix.lower() in image_extensions]

        for img_path in image_files:
            # Run inference
            results = model(img_path)
                                     
            # Get annotated image (BGR NumPy array)
            annotated_img = results[0].plot()

            # Append version to filename
            new_name = f"{img_path.stem}_{self.version}{img_path.suffix}"
            save_path = os.path.join(self.output_folder, new_name)

            # Save image
            cv2.imwrite(save_path, annotated_img)
            print(f"Inference done for {new_name} → saved to {self.output_folder}")     
                
        print(f"Verify model done.")

if __name__ == '__main__':
    verify_model = VerifyModel()
    args = verify_model.parse_arguments()
    verify_model.start(args)