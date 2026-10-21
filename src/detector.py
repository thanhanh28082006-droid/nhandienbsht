import cv2
import numpy as np
from ultralytics import YOLO
from src.config import setup_logger

logger = setup_logger("Detector")


class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.28):
        logger.info(f"Dang tai mo hinh Tim bien so (Stage 1): {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str):
        original_img = cv2.imread(image_path)
        if original_img is None:
            logger.error(f"Khong the doc anh tu: {image_path}")
            return None, [], None

        # Dung imgsz=1024 va agnostic_nms=True de bat duoc ca bien so o xa (nhu xe buyt 51B)
        results = self.model(
            original_img,
            imgsz=1024,
            conf=self.conf_threshold,
            iou=0.45,
            agnostic_nms=True,
            verbose=False
        )

        boxes_out = []
        annotated_img = original_img.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                if (x2 - x1) >= 15 and (y2 - y1) >= 10:
                    boxes_out.append([x1, y1, x2, y2, conf, cls])

        return annotated_img, boxes_out, original_img