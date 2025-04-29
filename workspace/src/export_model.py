import os
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

class ExportModel:
    def __init__(self):
        self.model_path = ""
        self.output_folder = ""
        self.format = "torchscript"


    def parse_arguments(self):
        parser = argparse.ArgumentParser(description = "Export YOLO Model.")
        parser.add_argument("--model_path", type=str, help="Path to the exported model.")
        parser.add_argument("--output_folder", type=str, help="Path to the folder to save exported model.")
        parser.add_argument("--format", default="torchscript", type=str, help="The format of the exported model. Default: torchscript")
        args = parser.parse_args()

        return args
    
    def get_latest_folder(folder_path):
        path = Path(folder_path)
        folders = [folder for folder in path.iterdir() if folder.is_dir()]
        latest_folder = max(folders, key=lambda x: x.stat().st_mtime)
        return str(latest_folder)

    def start(self, args):
        self.model_path = args.model_path or self.model_path
        self.output_folder = args.output_folder or self.output_folder
        self.format = args.format

        # Get latest folder
        if self.model_path == "":
            self.model_path = os.getcwd() + "/runs/detect/"
            latest_folder = ExportModel.get_latest_folder(self.model_path)
            self.model_path = os.path.join(latest_folder, "weights/best.pt")

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # export model
        model = YOLO(self.model_path)
        self.model_path = model.export(format=self.format)

        # move the model to output folder
        shutil.move(self.model_path, self.output_folder)

        print(f"Exported model saved to {self.output_folder}")

if __name__ == "__main__":
    export_model = ExportModel()
    export_model.start(export_model.parse_arguments())
