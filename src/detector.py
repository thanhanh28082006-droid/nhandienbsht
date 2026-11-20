import cv2
import numpy as np
from ultralytics import YOLO
from src.config import setup_logger

logger = setup_logger("Detector")


class YOLODetector:
    """
    STAGE 1 - BỘ PHÁT HIỆN VÙNG BIỂN SỐ TOÀN DIỆN (MULTI-ORIENTATION & MULTI-SCALE DETECTOR):
    1. Quét trên ảnh gốc (0°) ở đa độ phân giải (640, 800, 960).
    2. Quét trên các biến thể cân bằng sáng/tối (CLAHE LAB, Gamma nâng tối, Gamma giảm lóa)
       và quét bổ sung vùng nửa dưới ảnh (nơi biển số hay nằm sát mép như 29C-640.38).
    3. Quét chủ động 4 hướng xoay (0°, 90° CW, 90° CCW, 180°) và quy đổi tọa độ chính xác về ảnh gốc
       kèm nhãn hướng xoay 'orient' để Stage 2 xoay thẳng trước khi đọc OCR.
    4. Bộ đề xuất vùng ứng viên tương phản/cạnh (High-Contrast Candidate Proposal) dự phòng
       khi biển số bị lóa sáng mạnh hoặc sát mép ảnh.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.36):
        logger.info(f"Dang tai mo hinh Tim bien so (Stage 1): {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    @staticmethod
    def _adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / max(0.08, gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.float32)
        lut = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(image, lut)

    @staticmethod
    def _enhance_clahe_lab(image: np.ndarray, clip_limit: float = 2.8, grid: int = 8) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
        l_eq = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _map_box_to_original(rx1: float, ry1: float, rx2: float, ry2: float,
                             orig_w: int, orig_h: int, orient: str) -> tuple:
        if orient == "ROT90_CW":
            ox1, oy1 = ry1, orig_h - rx2
            ox2, oy2 = ry2, orig_h - rx1
        elif orient == "ROT90_CCW":
            ox1, oy1 = orig_w - ry2, rx1
            ox2, oy2 = orig_w - ry1, rx2
        elif orient == "ROT180":
            ox1, oy1 = orig_w - rx2, orig_h - ry2
            ox2, oy2 = orig_w - rx1, orig_h - ry1
        else:
            ox1, oy1, ox2, oy2 = rx1, ry1, rx2, ry2

        ox1 = max(0.0, min(float(orig_w - 1), float(ox1)))
        oy1 = max(0.0, min(float(orig_h - 1), float(oy1)))
        ox2 = max(0.0, min(float(orig_w - 1), float(ox2)))
        oy2 = max(0.0, min(float(orig_h - 1), float(oy2)))
        return ox1, oy1, ox2, oy2

    @staticmethod
    def _filter_valid_plate_boxes(boxes: list, img_w: int, img_h: int) -> list:
        valid = []
        for b in boxes:
            x1, y1, x2, y2, conf = b[:5]
            orient = b[5] if len(b) > 5 else "ROT0"

            bw = x2 - x1
            bh = y2 - y1
            if bw < 16 or bh < 12:
                continue

            aspect = bw / float(bh)
            # Cho phép cả biển ngang (0.65..4.6) và biển xoay dọc 90°/270° (0.20..1.65)
            # Cho phép từ 0.45 để bắt trọn biển xe máy chụp chéo/sát mép (như 59-X1 475.44)
            is_valid_ratio = (0.45 <= aspect <= 4.6) if orient == "ROT0" else (0.20 <= aspect <= 4.6)
            is_valid_size = (bw <= img_w * 0.88) and (bh <= img_h * 0.88)

            if is_valid_ratio and is_valid_size:
                valid.append([float(x1), float(y1), float(x2), float(y2), float(conf), orient])
        return valid

    @staticmethod
    def _nms_boxes(boxes: list, iou_thresh: float = 0.45) -> list:
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda x: x[4], reverse=True)
        kept = []
        for b in boxes_sorted:
            is_dup = False
            for k in kept:
                ix1 = max(b[0], k[0])
                iy1 = max(b[1], k[1])
                ix2 = min(b[2], k[2])
                iy2 = min(b[3], k[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                area_k = max(1.0, (k[2] - k[0]) * (k[3] - k[1]))
                union = area_b + area_k - inter
                if union > 0 and (inter / union) > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(b)
        return kept

    def _run_yolo_pass(self, img: np.ndarray, conf: float, imgsz: int = 640) -> list:
        found = []
        res = self.model(img, imgsz=imgsz, conf=conf, iou=0.45, verbose=False)
        if res and len(res) > 0 and res[0].boxes is not None:
            for box in res[0].boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf[0].cpu().numpy())
                found.append((float(rx1), float(ry1), float(rx2), float(ry2), c))
        return found

    @staticmethod
    def _propose_contrast_plate_candidates(image: np.ndarray) -> list:
        """
        Đề xuất vùng ứng viên biển số dựa trên mật độ biên ngang/dọc và tương phản cục bộ
        khi biển số bị lóa sáng mạnh hoặc nằm sát mép dưới ảnh (ví dụ 29C-640.38).
        """
        candidates = []
        try:
            h, w = image.shape[:2]
            if h < 40 or w < 40:
                return []

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            eq = clahe.apply(gray)

            # Toán tử hình thái học Black-Hat tìm chữ đen trên nền sáng (kể cả vùng lóa)
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            blackhat = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, rect_kern)

            grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
            grad_x = np.abs(grad_x)
            max_val = np.percentile(grad_x, 99.5) + 1e-5
            grad_x = np.clip((grad_x / max_val) * 255.0, 0, 255).astype(np.uint8)

            grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
            _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            close_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kern)
            closed = cv2.erode(closed, None, iterations=1)
            closed = cv2.dilate(closed, None, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_area = float(h * w)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = float(bw * bh)
                if area < img_area * 0.012 or area > img_area * 0.55:
                    continue
                if bw < 45 or bh < 16:
                    continue
                aspect = bw / float(bh)
                # Biển số 1 dòng hoặc 2 dòng nằm ngang
                if 0.85 <= aspect <= 4.6:
                    pad_x = int(bw * 0.14)
                    pad_y = int(bh * 0.22)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w - 1, x + bw + pad_x)
                    y2 = min(h - 1, y + bh + pad_y)
                    candidates.append([float(x1), float(y1), float(x2), float(y2), 0.32, "ROT0"])
        except Exception:
            pass
        return candidates[:4]

    def detect(self, image_path: str):
        original_img = cv2.imread(image_path)
        if original_img is None:
            return None, [], None

        h, w = original_img.shape[:2]
        raw_boxes = []

        # Tầng 1: Quét trên ảnh gốc (0°) với độ phân giải 640 và 800
        for sz in (640, 800):
            hits = self._run_yolo_pass(original_img, self.conf_threshold, imgsz=sz)
            for rx1, ry1, rx2, ry2, c in hits:
                ox1, oy1, ox2, oy2 = self._map_box_to_original(rx1, ry1, rx2, ry2, w, h, "ROT0")
                raw_boxes.append([ox1, oy1, ox2, oy2, c, "ROT0"])
            if raw_boxes:
                break

        # Tầng 2: Quét ảnh tăng cường sáng/giảm lóa và quét phóng đại vùng nửa dưới ảnh (trị biển sát mép)
        if not raw_boxes:
            enh_variants = [
                self._enhance_clahe_lab(original_img, clip_limit=2.8, grid=8),
                self._adjust_gamma(original_img, gamma=1.70),
                self._adjust_gamma(original_img, gamma=0.62),
            ]
            for variant_img in enh_variants:
                hits = self._run_yolo_pass(variant_img, max(0.22, self.conf_threshold - 0.14), imgsz=640)
                for rx1, ry1, rx2, ry2, c in hits:
                    ox1, oy1, ox2, oy2 = self._map_box_to_original(rx1, ry1, rx2, ry2, w, h, "ROT0")
                    raw_boxes.append([ox1, oy1, ox2, oy2, c, "ROT0"])
                if raw_boxes:
                    break

        if not raw_boxes and h > 120:
            # Quét riêng 60% nửa dưới khung hình (trị biển nằm sát cạnh đáy như 29C-640.38)
            y_offset = int(h * 0.40)
            bottom_roi = original_img[y_offset:h, 0:w]
            bottom_enh = self._enhance_clahe_lab(bottom_roi, clip_limit=3.0, grid=8)
            hits = self._run_yolo_pass(bottom_enh, 0.20, imgsz=640)
            for rx1, ry1, rx2, ry2, c in hits:
                raw_boxes.append([
                    float(rx1), float(ry1 + y_offset),
                    float(rx2), float(ry2 + y_offset),
                    c, "ROT0"
                ])

        # Tầng 3: Quét chủ động các hướng xoay 90° CW, 90° CCW, 180° (trị ảnh chụp dọc như 77-L1 600.30)
        if not raw_boxes:
            rot_configs = [
                ("ROT90_CW", cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)),
                ("ROT90_CCW", cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ("ROT180", cv2.rotate(original_img, cv2.ROTATE_180)),
            ]
            for orient_tag, rot_img in rot_configs:
                hits = self._run_yolo_pass(rot_img, max(0.22, self.conf_threshold - 0.14), imgsz=640)
                if not hits:
                    rot_enh = self._enhance_clahe_lab(rot_img, clip_limit=2.8, grid=8)
                    hits = self._run_yolo_pass(rot_enh, 0.20, imgsz=640)
                for rx1, ry1, rx2, ry2, c in hits:
                    ox1, oy1, ox2, oy2 = self._map_box_to_original(rx1, ry1, rx2, ry2, w, h, orient_tag)
                    raw_boxes.append([ox1, oy1, ox2, oy2, c, orient_tag])
                if raw_boxes:
                    break

        # Tầng 4: Đề xuất vùng ứng viên tương phản Black-Hat khi YOLO vẫn chưa bắt được
        if not raw_boxes:
            raw_boxes.extend(self._propose_contrast_plate_candidates(original_img))

        valid_boxes = self._filter_valid_plate_boxes(raw_boxes, w, h)
        final_boxes = self._nms_boxes(valid_boxes, iou_thresh=0.45)

        annotated_img = original_img.copy()
        return annotated_img, final_boxes, original_img
