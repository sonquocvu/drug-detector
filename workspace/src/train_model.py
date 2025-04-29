import argparse
import glob
import time
import os
from datetime import datetime
from ultralytics import YOLO

class TrainModel:

    def __init__(self):
        self.task = "detect"
        self.data = ""
        self.name = "training-model"
        self.epochs = 100
        self.img_size = 640
        self.batch_size = 16
        self.device = 0
        self.workers = 1
        self.patience = 5
        self.initial_learning_rate = 0.01
        self.optimizer = "AdamW"
        self.verbose = False

    def parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Training a model using YOLO.")
        parser.add_argument(
            "--data",
            type=str,
            help="Path to the data file (.yml file)."
        )
        parser.add_argument(
            "--name",
            type=str,
            help="Name of the model."
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="AdamW",
            help="The optimizer (SGD, Adam, AdamW). Default: AmdaW."
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=50,
            help="The number of epoch to train model. Default: 50"
        )
        parser.add_argument(
            "--imgsz",
            type=int,
            default=640,
            help="The size of labeled images to train model. Default: 640."
        )
        parser.add_argument(
            "--batch",
            type=int,
            default=16,
            help="The number of images are handled at the same time. Default: 16."
        )
        parser.add_argument(
            "--device",
            type=int,
            default=0,
            help="The GPU's identity. Default: 0."
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="The number of CPU cores use. Default: 4"
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="The number of epock to wait for new improvement. Default: 5"
        )
        parser.add_argument(
            "--lrt",
            type=float,
            help="The initial learning rate. Default: 0.01."
        )
        parser.add_argument(
            "--verbose",
            action='store_true',
            help="To enable showing log."
        )
        args = parser.parse_args()
        return args
    
    def start(self, args):
        self.data = args.data or self.data
        self.name = args.name or self.name
        self.epochs = args.epochs or self.epochs
        self.img_size = args.imgsz or self.img_size
        self.batch_size = args.batch or self.batch_size
        self.device = args.device or self.device
        self.workers = args.workers or self.workers
        self.patience = args.patience or self.patience
        self.optimizer = args.optimizer or self.optimizer
        self.initial_learning_rate = args.lrt or self.initial_learning_rate * (self.batch_size / 64)
        self.verbose = args.verbose

        # Load a YOLOv8 model
        model = YOLO('yolov8m.pt')

        # Train the model
        model.train(
            task=self.task,
            data=self.data,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            optimizer=self.optimizer,
            lr0=self.initial_learning_rate,
            device=self.device,
            name=self.name,
            workers=self.workers,
            patience=self.patience,
            verbose=self.verbose,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.15,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.0,            
        )

if __name__ == "__main__":
    training_model = TrainModel()
    args = training_model.parse_arguments()
    training_model.start(args)