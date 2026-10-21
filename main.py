import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import psycopg2
import random
from datetime import datetime, timedelta

from src.config import TEST_IMAGES_DIR, MODELS_DIR, BASE_DIR, get_province_name, setup_logger
from src.detector import YOLODetector
from src.ocr_engine import OCREngine

logger = setup_logger("MasterSystem")


class ALPRSystem:
    def __init__(self):
        logger.info("=" * 60)
        logger.info(" HE THONG ALPR YOLOV11 - POSTGRESQL DATABASE PRO")
        logger.info("=" * 60)

        detector_path = MODELS_DIR / "yolo_detector.pt"
        self.detector = YOLODetector(model_path=str(detector_path), conf_threshold=0.45)
        self.ocr = OCREngine()

        self.db_params = {
            "host": "localhost",
            "database": "nhandien_bienso",
            "user": "postgres",
            "password": "Hoangthanh.08",
            "port": "5432"
        }

    def query_database(self, plate_text):
        if not plate_text:
            return {"owner": "---", "reg_date": "---", "status": "---"}

        clean_plate = "".join(filter(str.isalnum, plate_text))

        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            cursor.execute("SELECT owner_name, reg_date, status FROM public.vehicles WHERE plate_number=%s",
                           (clean_plate,))
            result = cursor.fetchone()

            if result:
                conn.close()
                return {"owner": result[0], "reg_date": result[1], "status": result[2]}
            else:
                first_names = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Do"]
                last_names = ["A", "B", "Hung", "Lan", "Kien", "Toan", "Phuc", "Thanh", "Minh", "Thu"]
                random_owner = f"{random.choice(first_names)} Van {random.choice(last_names)}"

                random_days_ago = random.randint(100, 1800)
                random_date = (datetime.now() - timedelta(days=random_days_ago)).strftime("%d/%m/%Y")

                statuses = ["HOP LE", "HOP LE", "HOP LE", "PHAT NGUOI", "HET HAN DANG KIEM", "TRUY NA"]
                random_status = random.choice(statuses)

                cursor.execute('''
                               INSERT INTO public.vehicles (plate_number, owner_name, reg_date, status)
                               VALUES (%s, %s, %s, %s)
                               ''', (clean_plate, random_owner, random_date, random_status))

                conn.commit()
                conn.close()

                return {"owner": random_owner, "reg_date": random_date, "status": random_status}

        except Exception as e:
            logger.error(f"Loi ket noi PostgreSQL: {e}")
            return {"owner": "Loi Ket Noi DB", "reg_date": "---", "status": "---"}

    def smart_crop(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin_ratio: float = 0.06) -> np.ndarray:
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * margin_ratio), int(bh * margin_ratio)

        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        nx2, ny2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        return img[ny1:ny2, nx1:nx2]

    def process_image(self, image_path: Path):
        logger.info(f"\n[>] Dang phan tich: {image_path.name}")
        annotated_img, boxes_out, original_img = self.detector.detect(str(image_path))
        if original_img is None:
            return

        clean_img = original_img.copy()
        h, w = clean_img.shape[:2]
        panel_w = 460

        dash_h = max(h, 850)
        dashboard = np.zeros((dash_h, w + panel_w, 3), dtype=np.uint8)
        dashboard[:] = (20, 20, 20)

        dashboard[:h, :w] = clean_img
        cv2.rectangle(dashboard, (w, 0), (w + panel_w, dash_h), (30, 30, 30), -1)

        cv2.putText(dashboard, "HE THONG ALPR & POSTGRESQL", (w + 20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255),
                    2, cv2.LINE_AA)
        cv2.line(dashboard, (w + 20, 55), (w + panel_w - 20, 55), (100, 100, 100), 2)

        if not boxes_out:
            cv2.putText(dashboard, "[X] KHONG TIM THAY BIEN SO", (w + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        else:
            boxes_out.sort(key=lambda b: b[0])

            valid_boxes = []
            for box in boxes_out:
                if len(box) < 4: continue
                x1, y1, x2, y2 = map(int, box[:4])
                bw = x2 - x1
                bh = y2 - y1
                if bh == 0: continue

                aspect_ratio = bw / float(bh)

                if aspect_ratio < 0.8 or aspect_ratio > 3.5 or bw > (w * 0.4):
                    continue
                valid_boxes.append(box)

            num_plates = len(valid_boxes)
            y_offset = 70

            if num_plates == 0:
                cv2.putText(dashboard, "[X] KHONG CO BIEN SO HOP LE", (w + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            else:
                avail_h_per_plate = (dash_h - 80) // max(1, min(num_plates, 3))
                max_img_h = max(90, avail_h_per_plate - 170)
                max_img_w = panel_w - 40

                for idx, box in enumerate(valid_boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    plate_conf = int(box[4] * 100) if len(box) >= 5 else 0

                    cropped_plate = self.smart_crop(original_img, x1, y1, x2, y2, margin_ratio=0.06)
                    if cropped_plate.size == 0:
                        continue

                    bien_so_text, traced_img, avg_conf = self.ocr.extract_text(cropped_plate)

                    tinh_thanh = get_province_name(bien_so_text)
                    db_info = self.query_database(bien_so_text)

                    if plate_conf == 0:
                        plate_conf = int(avg_conf)

                    status = db_info["status"]
                    alert_color = (0, 0, 255) if status in ["PHAT NGUOI", "HET HAN DANG KIEM", "TRUY NA",
                                                            "CHUA DANG KY"] else (0, 255, 255)

                    cv2.rectangle(dashboard, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    text_top = f"Bien so ({plate_conf}%)"
                    (tw_t, th_t), _ = cv2.getTextSize(text_top, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    top_box_y1 = max(0, y1 - th_t - 10)
                    cv2.rectangle(dashboard, (x1, top_box_y1), (x1 + tw_t + 10, y1), (0, 0, 0), -1)
                    cv2.putText(dashboard, text_top, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                                cv2.LINE_AA)

                    display_ocr = bien_so_text if bien_so_text else "Dang doc..."
                    text_bot = f"OCR: {display_ocr} | {tinh_thanh}"
                    (tw_b, th_b), _ = cv2.getTextSize(text_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(dashboard, (x1, y2), (x1 + tw_b + 10, min(dash_h, y2 + th_b + 12)), (0, 0, 0), -1)
                    cv2.putText(dashboard, text_bot, (x1 + 5, min(dash_h - 4, y2 + th_b + 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, alert_color, 2, cv2.LINE_AA)

                    th, tw = traced_img.shape[:2]
                    if tw > 0 and th > 0:
                        scale = min(max_img_w / float(tw), max_img_h / float(th))
                        new_w, new_h = int(tw * scale), int(th * scale)

                        if y_offset + new_h + 140 <= dash_h:
                            traced_resized = cv2.resize(traced_img, (new_w, new_h))
                            dashboard[y_offset:y_offset + new_h, w + 20:w + 20 + new_w] = traced_resized
                            cv2.rectangle(dashboard, (w + 20, y_offset), (w + 20 + new_w, y_offset + new_h),
                                          (0, 255, 0), 2)

                            info_y = y_offset + new_h + 25
                            cv2.putText(dashboard, f"OCR #{idx + 1}: {bien_so_text} ({avg_conf:.0f}%)",
                                        (w + 20, info_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(dashboard, f"TINH: {tinh_thanh}", (w + 20, info_y + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(dashboard, f"Chu xe: {db_info['owner']}", (w + 20, info_y + 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(dashboard, f"Ngay DK: {db_info['reg_date']}", (w + 20, info_y + 75),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(dashboard, f"TT: {db_info['status']}", (w + 20, info_y + 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, alert_color, 2, cv2.LINE_AA)
                            cv2.line(dashboard, (w + 20, info_y + 120), (w + panel_w - 20, info_y + 120), (80, 80, 80),
                                     1)
                            y_offset = info_y + 135

        cv2.imshow("He Thong Canh Bao ALPR & PostgreSQL", dashboard)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_application(app: ALPRSystem):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = TEST_IMAGES_DIR if any(TEST_IMAGES_DIR.glob("*")) else BASE_DIR
    file_path = filedialog.askopenfilename(
        parent=root, initialdir=str(initial_dir),
        title="Chon anh bien so xe (JPG/PNG)", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    if file_path:
        app.process_image(Path(file_path))


if __name__ == "__main__":
    app = ALPRSystem()
    run_application(app)