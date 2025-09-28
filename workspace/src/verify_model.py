import argparse
import os
import json
from pathlib import Path
from ultralytics import YOLO


class VerifyModel():

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description = "Test YOLO Model on an Image.")
        parser.add_argument("--engine", required=True, help="Path to .engine file")
        parser.add_argument("--imgsz", type=int, default=640, help="Inference size; must match engine build (e.g., 640 or 832)")
        parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
        parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
        parser.add_argument("--source", required=True, help="Image path, folder path, video path, or webcam index (e.g., 0)")
        parser.add_argument("--out", default="runs_detect_trt", help="Output directory")
        args = parser.parse_args()

        return args
    
    def start(self, args):

        # Make sure output folder exists
        os.makedirs(args.out, exist_ok=True)

        # Load model
        model = YOLO(args.engine, task="detect")

        # Map to nice names
        code_to_nice = {}
        HERE = Path(__file__).parent
        json_path = HERE / "drug_name.json"
        with json_path.open("r", encoding="utf-8") as f:
            code_to_nice = json.load(f)
        
        id_to_nice = {
            cid: code_to_nice.get(name, name) for cid, name in model.names.items()
        }
        
        # Try to set on the underlying model first
        if hasattr(model, "model") and hasattr(model.model, "names"):
            print("Debug 1")
            model.model.names = id_to_nice  # <- usually works

        # Some versions also read from predictor.names
        if hasattr(model, "predictor"):
            try:
                print("Debug 2")
                model.predictor.names = id_to_nice
            except Exception:
                print("Debug 3")
                pass

        # Run inference
        model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=0,
            save=True,
            project=args.out,
            name="yolo8s_results",
            verbose=False
        )
                                          
        print(f"Verify model done.")

if __name__ == '__main__':
    verify_model = VerifyModel()
    args = verify_model.parse_arguments()
    verify_model.start(args)