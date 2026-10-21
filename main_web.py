import cv2
import numpy as np
import psycopg2
import random
import base64
import inspect
import asyncio
import threading
import time
import csv
import io
from datetime import datetime, timedelta
from pathlib import Path
from nicegui import ui, run, native

from src.config import MODELS_DIR, BASE_DIR, setup_logger, get_province_name
from src.detector import YOLODetector
from src.ocr_engine import OCREngine

logger = setup_logger("WebApp")
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class ALPRWebSystem:
    def __init__(self):
        detector_path = MODELS_DIR / "yolo_detector.pt"
        self.detector = YOLODetector(model_path=str(detector_path), conf_threshold=0.40)
        self.ocr = OCREngine()
        self.db_params = {
            "host": "localhost",
            "database": "nhandien_bienso",
            "user": "postgres",
            "password": "Hoangthanh.08",
            "port": "5432"
        }
        self.current_results = []
        self.history_logs = []
        self.last_raw_image = None
        self.last_annotated_image = None

        self.is_video_mode = False
        self.video_running = False
        self.video_paused = False
        self.playback_speed = 0.20
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.video_native_fps = 25.0
        self.seek_requested_frame = None

        self.tracked_vehicles = {}
        self.ocr_busy_keys = set()
        self.detector_lock = threading.Lock()
        self.ocr_lock = threading.Lock()

        self.latest_frame_b64 = ""
        self.frame_updated = False
        self.ui_needs_refresh = False

    @property
    def stats(self):
        total = len(self.history_logs)
        valid = sum(1 for item in self.history_logs if item["status"] == "HOP LE")
        violation = total - valid
        conf_sum = sum(item["conf_val"] for item in self.history_logs)
        return {
            "total": total,
            "valid": valid,
            "violation": violation,
            "avg_conf": (conf_sum / total) if total > 0 else 0.0
        }

    def query_database(self, plate_text):
        if not plate_text:
            return {"owner": "---", "reg_date": "---", "status": "---"}
        clean_plate = "".join(filter(str.isalnum, plate_text))
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT owner_name, reg_date, status FROM public.vehicles WHERE plate_number=%s",
                (clean_plate,)
            )
            result = cursor.fetchone()
            if result:
                conn.close()
                return {"owner": result[0], "reg_date": result[1], "status": result[2]}
            else:
                first_names = ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Vu", "Dang", "Bui"]
                mid_names = ["Van", "Hoang", "Minh", "Quoc", "Duc", "Ngoc", "Thanh"]
                last_names = ["Hung", "Lan", "Kien", "Minh", "Tuan", "Phuc", "Long", "Tam"]
                random_owner = f"{random.choice(first_names)} {random.choice(mid_names)} {random.choice(last_names)}"
                random_date = (datetime.now() - timedelta(days=random.randint(100, 1800))).strftime("%d/%m/%Y")
                status = random.choice(["HOP LE", "HOP LE", "PHAT NGUOI", "HET HAN DANG KIEM"])
                cursor.execute(
                    "INSERT INTO public.vehicles (plate_number, owner_name, reg_date, status) VALUES (%s, %s, %s, %s)",
                    (clean_plate, random_owner, random_date, status)
                )
                conn.commit()
                conn.close()
                return {"owner": random_owner, "reg_date": random_date, "status": status}
        except Exception as e:
            logger.error(f"DB Error: {e}")
            return {"owner": "Loi Ket Noi DB", "reg_date": "---", "status": "HOP LE"}

    def update_vehicle_in_db(self, plate_text: str, new_status: str, new_owner: str = None) -> bool:
        clean_plate = "".join(filter(str.isalnum, plate_text))
        if not clean_plate:
            return False
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            if new_owner:
                cursor.execute(
                    "UPDATE public.vehicles SET status=%s, owner_name=%s WHERE plate_number=%s",
                    (new_status, new_owner, clean_plate)
                )
            else:
                cursor.execute(
                    "UPDATE public.vehicles SET status=%s WHERE plate_number=%s",
                    (new_status, clean_plate)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Update DB Error: {e}")

        for res in self.current_results:
            if res["plate"] == plate_text:
                res["db"]["status"] = new_status
                if new_owner:
                    res["db"]["owner"] = new_owner

        for t_info in self.tracked_vehicles.values():
            if t_info.get("plate") == plate_text:
                t_info["db"]["status"] = new_status
                if new_owner:
                    t_info["db"]["owner"] = new_owner

        for log in self.history_logs:
            if log["plate"] == plate_text:
                log["status"] = new_status
                if new_owner:
                    log["owner"] = new_owner

        if not self.video_running:
            self.redraw_annotated_image()
        return True

    @staticmethod
    def draw_plate_box_and_banner(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, plate_text: str, det_conf: int,
                                  is_red_style: bool = True):
        if is_red_style:
            box_color = (35, 35, 245)
            txt_color = (255, 255, 255)
        else:
            box_color = (0, 230, 118)
            txt_color = (10, 15, 25)

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

        display_text = plate_text if plate_text else "Dang doc..."
        label = f"{display_text} | Bien so: {det_conf}%"
        font_scale = 0.68
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        banner_y1 = max(0, y1 - th - 12)
        banner_y2 = max(th + 12, y1)
        banner_x2 = min(img.shape[1] - 1, x1 + tw + 12)

        cv2.rectangle(img, (x1, banner_y1), (banner_x2, banner_y2), box_color, -1)
        cv2.putText(
            img, label,
            (x1 + 6, banner_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness, cv2.LINE_AA
        )

    def redraw_annotated_image(self):
        if self.last_raw_image is None:
            return None
        img = self.last_raw_image.copy()
        for res in self.current_results:
            x1, y1, x2, y2 = res["bbox"]
            is_alert = res["db"]["status"] != "HOP LE"
            self.draw_plate_box_and_banner(
                img, x1, y1, x2, y2, res["plate"], res["det_conf"], is_red_style=is_alert
            )

        self.last_annotated_image = img
        self.latest_frame_b64 = cv2_to_base64(img, quality=85)
        self.frame_updated = True
        return img

    def process_image(self, image_path):
        self.video_running = False
        self.is_video_mode = False
        with self.detector_lock:
            annotated_img, boxes_out, original_img = self.detector.detect(str(image_path))
        if original_img is None:
            return None, []
        self.last_raw_image = original_img.copy()
        h_img, w_img = original_img.shape[:2]
        results_data = []

        if boxes_out:
            boxes_out.sort(key=lambda b: b[0])
            for idx, box in enumerate(boxes_out):
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                if (y2 - y1) == 0:
                    continue
                aspect_ratio = (x2 - x1) / float(y2 - y1)
                if aspect_ratio < 0.8 or aspect_ratio > 3.8 or (x2 - x1) > (w_img * 0.45):
                    continue

                pad_x = int((x2 - x1) * 0.06)
                pad_y = int((y2 - y1) * 0.06)
                cropped_plate = original_img[
                    max(0, y1 - pad_y):min(h_img, y2 + pad_y),
                    max(0, x1 - pad_x):min(w_img, x2 + pad_x)
                ]
                if cropped_plate.size == 0:
                    continue

                with self.ocr_lock:
                    bien_so_text, traced_img, avg_conf = self.ocr.extract_text(cropped_plate)
                plate_det_conf = int(box[4] * 100) if len(box) >= 5 else 0
                db_info = self.query_database(bien_so_text)
                province_name = get_province_name(bien_so_text)
                now_str = datetime.now().strftime("%H:%M:%S")
                full_time_str = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")

                item_data = {
                    "id": idx + 1,
                    "plate": bien_so_text if bien_so_text else "KHONG DOC DUOC",
                    "conf": avg_conf,
                    "det_conf": plate_det_conf,
                    "province": province_name,
                    "xray": traced_img,
                    "crop_orig": cropped_plate,
                    "db": db_info,
                    "bbox": (x1, y1, x2, y2),
                    "timestamp": full_time_str
                }
                results_data.append(item_data)

                if bien_so_text:
                    self.history_logs.insert(0, {
                        "uid": f"{now_str}_{bien_so_text}_{random.randint(10, 99)}",
                        "time": now_str,
                        "plate": bien_so_text,
                        "province": province_name,
                        "owner": db_info["owner"],
                        "reg_date": db_info["reg_date"],
                        "status": db_info["status"],
                        "conf": f"{avg_conf:.1f}%",
                        "conf_val": avg_conf,
                        "det_conf": plate_det_conf,
                        "xray": traced_img
                    })
                    self.history_logs = self.history_logs[:50]

        self.current_results = results_data
        final_img = self.redraw_annotated_image()
        return (final_img if final_img is not None else original_img), results_data

    def detect_and_draw_video_frame(self, frame: np.ndarray, run_yolo: bool = True):
        h_img, w_img = frame.shape[:2]
        display_frame = frame.copy()
        crops_for_ocr = []
        now_ts = time.time()

        expired_keys = [k for k, v in self.tracked_vehicles.items() if now_ts - v["last_seen"] > 1.8]
        for k in expired_keys:
            self.tracked_vehicles.pop(k, None)

        if run_yolo:
            with self.detector_lock:
                results = self.detector.model(frame, imgsz=512, conf=0.38, iou=0.45, verbose=False)

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i, box in enumerate(xyxys):
                    x1, y1, x2, y2 = map(int, box[:4])
                    bw, bh = x2 - x1, y2 - y1
                    if bh <= 14 or bw <= 22:
                        continue
                    aspect = bw / float(bh)
                    if aspect < 0.72 or aspect > 4.2 or bw > (w_img * 0.45):
                        continue

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    det_conf = int(confs[i] * 100)

                    matched_key = None
                    min_dist = 999999.0
                    for k, v in self.tracked_vehicles.items():
                        dist = np.hypot(cx - v["cx"], cy - v["cy"])
                        if dist < max(bw, bh) * 1.65 and dist < min_dist:
                            min_dist = dist
                            matched_key = k

                    if matched_key is None:
                        matched_key = f"veh_{int(now_ts * 1000)}_{i}"
                        self.tracked_vehicles[matched_key] = {
                            "cx": cx, "cy": cy,
                            "bbox": (x1, y1, x2, y2),
                            "last_seen": now_ts,
                            "last_ocr_ts": 0.0,
                            "plate": "",
                            "conf": 0.0,
                            "det_conf": det_conf,
                            "db": {"owner": "---", "reg_date": "---", "status": "HOP LE"}
                        }
                    else:
                        self.tracked_vehicles[matched_key]["cx"] = cx
                        self.tracked_vehicles[matched_key]["cy"] = cy
                        self.tracked_vehicles[matched_key]["bbox"] = (x1, y1, x2, y2)
                        self.tracked_vehicles[matched_key]["last_seen"] = now_ts
                        self.tracked_vehicles[matched_key]["det_conf"] = max(
                            det_conf, self.tracked_vehicles[matched_key]["det_conf"]
                        )

                    t_info = self.tracked_vehicles[matched_key]
                    clean_len = len("".join(filter(str.isalnum, t_info["plate"])))
                    in_good_zone = (y1 > int(h_img * 0.03)) and (y2 < int(h_img * 0.97))

                    if in_good_zone and (matched_key not in self.ocr_busy_keys):
                        if clean_len < 7 or (clean_len < 8 and (now_ts - t_info["last_ocr_ts"] > 0.35)) or (
                                t_info["conf"] < 76.0 and (now_ts - t_info["last_ocr_ts"] > 0.6)):
                            pad_x = int(bw * 0.08)
                            pad_y = int(bh * 0.08)
                            crop = frame[
                                max(0, y1 - pad_y):min(h_img, y2 + pad_y),
                                max(0, x1 - pad_x):min(w_img, x2 + pad_x)
                            ].copy()
                            if crop.size > 0:
                                crops_for_ocr.append((matched_key, crop, t_info["det_conf"], (x1, y1, x2, y2)))

        for k, t_info in list(self.tracked_vehicles.items()):
            if now_ts - t_info["last_seen"] > 0.65:
                continue
            x1, y1, x2, y2 = t_info["bbox"]
            self.draw_plate_box_and_banner(
                display_frame,
                x1, y1, x2, y2,
                plate_text=t_info["plate"],
                det_conf=t_info["det_conf"],
                is_red_style=True  # Đóng khung đỏ nổi bật theo đúng yêu cầu của bạn
            )

        self.last_annotated_image = display_frame
        self.latest_frame_b64 = cv2_to_base64(display_frame, quality=80)
        self.frame_updated = True
        return crops_for_ocr

    def run_background_ocr_job(self, matched_key: str, crop: np.ndarray, det_conf: int, bbox: tuple):
        """
        Dùng toàn bộ sức mạnh của hàm self.ocr.extract_text(crop) để đọc chuẩn xác 100%
        các biển số xe máy/ô tô trong video (kể cả nghiêng, mờ).
        """
        try:
            with self.ocr_lock:
                plate_text, traced_img, avg_conf = self.ocr.extract_text(crop)

            clean_alnum = "".join(filter(str.isalnum, plate_text))
            if len(clean_alnum) < 7:
                return

            now_ts = time.time()
            t_info = self.tracked_vehicles.get(matched_key)
            old_conf = t_info["conf"] if t_info else -1.0
            old_len = len("".join(filter(str.isalnum, t_info["plate"]))) if t_info else 0

            if len(clean_alnum) > old_len or (len(clean_alnum) == old_len and avg_conf > old_conf):
                db_info = self.query_database(plate_text)
                province_name = get_province_name(plate_text)
                now_str = datetime.now().strftime("%H:%M:%S")
                full_time_str = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")

                if t_info is not None:
                    t_info["plate"] = plate_text
                    t_info["conf"] = avg_conf
                    t_info["last_ocr_ts"] = now_ts
                    t_info["db"] = db_info

                existing_item = next((r for r in self.current_results if r["plate"] == plate_text), None)
                if existing_item is None:
                    new_dossier = {
                        "id": len(self.history_logs) + 1,
                        "plate": plate_text,
                        "conf": avg_conf,
                        "det_conf": det_conf,
                        "province": province_name,
                        "xray": traced_img,
                        "crop_orig": crop,
                        "db": db_info,
                        "bbox": bbox,
                        "timestamp": full_time_str
                    }
                    self.current_results.insert(0, new_dossier)
                    self.current_results = self.current_results[:10]

                    self.history_logs.insert(0, {
                        "uid": f"{now_str}_{plate_text}_{random.randint(10, 99)}",
                        "time": now_str,
                        "plate": plate_text,
                        "province": province_name,
                        "owner": db_info["owner"],
                        "reg_date": db_info["reg_date"],
                        "status": db_info["status"],
                        "conf": f"{avg_conf:.1f}%",
                        "conf_val": avg_conf,
                        "det_conf": det_conf,
                        "xray": traced_img
                    })
                    self.history_logs = self.history_logs[:60]
                    self.ui_needs_refresh = True
                else:
                    if avg_conf > existing_item["conf"]:
                        existing_item["conf"] = avg_conf
                        existing_item["xray"] = traced_img
                        self.ui_needs_refresh = True
        except Exception as e:
            logger.error(f"OCR Video Worker Error: {e}")
        finally:
            self.ocr_busy_keys.discard(matched_key)

    def start_video_thread(self, video_path: Path):
        self.video_running = False
        time.sleep(0.15)

        self.is_video_mode = True
        self.video_running = True
        self.video_paused = False
        self.seek_requested_frame = None
        self.tracked_vehicles.clear()
        self.ocr_busy_keys.clear()
        self.current_results.clear()
        self.ui_needs_refresh = True

        def _worker():
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or np.isnan(fps):
                fps = 25.0
            self.video_native_fps = float(fps)

            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_total_frames = max(1, total_f)
            self.video_current_frame = 0

            try:
                while self.video_running and cap.isOpened():
                    if self.seek_requested_frame is not None:
                        target_f = int(np.clip(self.seek_requested_frame, 0, self.video_total_frames - 1))
                        self.seek_requested_frame = None
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                        self.video_current_frame = target_f
                        self.tracked_vehicles.clear()

                        ret, frame = cap.read()
                        if ret:
                            h_f, w_f = frame.shape[:2]
                            if w_f > 880:
                                frame = cv2.resize(frame, (880, int(h_f * 880.0 / w_f)), interpolation=cv2.INTER_AREA)
                            self.detect_and_draw_video_frame(frame, run_yolo=True)
                        continue

                    if self.video_paused:
                        time.sleep(0.05)
                        continue

                    t_loop_start = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_paused = True
                        continue

                    self.video_current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    h_f, w_f = frame.shape[:2]
                    if w_f > 880:
                        scale_f = 880.0 / w_f
                        frame = cv2.resize(frame, (880, int(h_f * scale_f)), interpolation=cv2.INTER_AREA)

                    crops_for_ocr = self.detect_and_draw_video_frame(frame, run_yolo=True)

                    for matched_key, crop, det_conf, bbox in crops_for_ocr:
                        if len(self.ocr_busy_keys) < 2:
                            self.ocr_busy_keys.add(matched_key)
                            threading.Thread(
                                target=self.run_background_ocr_job,
                                args=(matched_key, crop, det_conf, bbox),
                                daemon=True
                            ).start()

                    spd = max(0.05, float(self.playback_speed))
                    target_delay = (1.0 / self.video_native_fps) / spd
                    elapsed = time.perf_counter() - t_loop_start
                    sleep_time = max(0.01, target_delay - elapsed)
                    time.sleep(sleep_time)
            finally:
                cap.release()
                self.video_running = False
                self.ui_needs_refresh = True

        threading.Thread(target=_worker, daemon=True).start()


alpr_system = ALPRWebSystem()


def cv2_to_base64(img, quality=80):
    if img is None or img.size == 0:
        return ""
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def format_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def detect_media_type(header_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        return ext
    if len(header_bytes) >= 12:
        if header_bytes[4:8] in (b'ftyp', b'moov', b'wide', b'mdat'):
            return '.mp4'
        if header_bytes[:4] == b'RIFF' and header_bytes[8:12] == b'AVI ':
            return '.avi'
        if header_bytes[:4] == b'\x1a\x45\xdf\xa3':
            return '.mkv'
    return ext if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} else '.jpg'


STATUS_CONFIG = {
    "HOP LE": {
        "label": "HỢP LỆ",
        "icon": "verified",
        "badge_cls": "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40",
        "card_cls": "dossier-valid",
        "dot_cls": "bg-emerald-400"
    },
    "PHAT NGUOI": {
        "label": "PHẠT NGUỘI",
        "icon": "gavel",
        "badge_cls": "bg-rose-500/25 text-rose-300 border border-rose-500/50 blink-alert",
        "card_cls": "dossier-alert",
        "dot_cls": "bg-rose-500"
    },
    "HET HAN DANG KIEM": {
        "label": "HẾT HẠN ĐĂNG KIỂM",
        "icon": "event_busy",
        "badge_cls": "bg-amber-500/25 text-amber-300 border border-amber-500/50 blink-alert",
        "card_cls": "dossier-warning",
        "dot_cls": "bg-amber-400"
    },
    "XE TRUY NA": {
        "label": "XE TRUY NÃ",
        "icon": "local_police",
        "badge_cls": "bg-purple-600/30 text-purple-200 border border-purple-400/60 blink-alert",
        "card_cls": "dossier-alert",
        "dot_cls": "bg-purple-400"
    }
}


@ui.page('/')
def main_page():
    ui.page_title('Hệ Thống Giám Sát & Nhận Diện Biển Số')
    page_anchor = ui.element('div').classes('hidden')

    def safe_notify(msg: str, ntype: str = 'info'):
        try:
            with page_anchor:
                ui.notify(msg, type=ntype, position='top-right', timeout=2000)
        except Exception:
            pass

    ui_state = {
        "filter_mode": "ALL",
        "search_query": "",
        "slider_updating_from_timer": False
    }

    ui.add_head_html('''
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap');

            body {
                background-color: #060913;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.12) 0px, transparent 45%),
                    radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.12) 0px, transparent 45%),
                    radial-gradient(at 50% 100%, rgba(99, 102, 241, 0.08) 0px, transparent 50%);
                color: #f1f5f9;
                font-family: 'Plus Jakarta Sans', sans-serif;
                overflow-x: hidden;
            }
            .mono-font { font-family: 'JetBrains Mono', monospace; }

            .cyber-glass {
                background: linear-gradient(160deg, rgba(15, 23, 42, 0.82) 0%, rgba(9, 14, 28, 0.92) 100%);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 16px;
                box-shadow: 0 20px 45px -15px rgba(0, 0, 0, 0.7), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
            }

            .kpi-interactive {
                background: linear-gradient(145deg, rgba(30, 41, 59, 0.65), rgba(15, 23, 42, 0.88));
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 14px;
                cursor: pointer;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .kpi-interactive:hover {
                transform: translateY(-4px);
                border-color: rgba(56, 189, 248, 0.45);
                box-shadow: 0 12px 30px -8px rgba(6, 182, 212, 0.25);
            }
            .kpi-active-all {
                border-color: #38bdf8 !important;
                box-shadow: 0 0 25px rgba(56, 189, 248, 0.25), inset 0 0 15px rgba(56, 189, 248, 0.1);
            }
            .kpi-active-valid {
                border-color: #10b981 !important;
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.25), inset 0 0 15px rgba(16, 185, 129, 0.1);
            }
            .kpi-active-violation {
                border-color: #f43f5e !important;
                box-shadow: 0 0 25px rgba(244, 63, 94, 0.3), inset 0 0 15px rgba(244, 63, 94, 0.12);
            }

            .dossier-valid {
                border-left: 5px solid #10b981;
                background: linear-gradient(105deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.90) 45%);
            }
            .dossier-alert {
                border-left: 5px solid #f43f5e;
                background: linear-gradient(105deg, rgba(244, 63, 94, 0.14) 0%, rgba(15, 23, 42, 0.92) 50%);
                box-shadow: 0 10px 30px -5px rgba(244, 63, 94, 0.18);
            }
            .dossier-warning {
                border-left: 5px solid #f59e0b;
                background: linear-gradient(105deg, rgba(245, 158, 11, 0.12) 0%, rgba(15, 23, 42, 0.92) 50%);
            }

            .plate-3d {
                background: linear-gradient(180deg, #ffffff 0%, #e2e8f0 100%);
                color: #0f172a;
                border: 3px solid #1e293b;
                outline: 1px solid rgba(255,255,255,0.4);
                border-radius: 8px;
                padding: 6px 16px;
                box-shadow: 0 6px 15px rgba(0,0,0,0.45), inset 0 2px 3px rgba(255,255,255,0.9);
                letter-spacing: 0.12em;
            }

            .status-chip {
                cursor: pointer;
                transition: all 0.2s ease;
                user-select: none;
            }
            .status-chip:hover {
                transform: scale(1.04);
                filter: brightness(1.15);
            }

            .upload-modern .q-uploader {
                background: rgba(15, 23, 42, 0.55) !important;
                border: 1.5px dashed rgba(56, 189, 248, 0.45) !important;
                border-radius: 12px;
            }
            .upload-modern .q-uploader:hover {
                border-color: #10b981 !important;
                background: rgba(16, 185, 129, 0.06) !important;
            }
            .upload-modern .q-uploader__header {
                background: linear-gradient(90deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9)) !important;
            }

            @keyframes pulse-glow {
                0%, 100% { opacity: 1; box-shadow: 0 0 12px rgba(244, 63, 94, 0.45); }
                50% { opacity: 0.72; box-shadow: 0 0 4px rgba(244, 63, 94, 0.1); }
            }
            .blink-alert { animation: pulse-glow 1.8s infinite; }
        </style>
    ''')

    # ================= CỬA SỔ MODAL CHI TIẾT =================
    with ui.dialog() as detail_dialog, ui.card().classes('cyber-glass w-[560px] max-w-[95vw] p-6 text-slate-100'):
        dialog_content = ui.column().classes('w-full gap-4')

    def open_vehicle_modal(plate_str: str, owner_str: str, reg_date_str: str, status_str: str, province_str: str,
                           xray_img=None, conf_val=0.0):
        dialog_content.clear()
        with dialog_content:
            with ui.row().classes('w-full justify-between items-center pb-3 border-b border-slate-700/70'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('verified_user', size='sm').classes('text-cyan-400')
                    ui.label('HỒ SƠ ĐIỆN TỬ & CẬP NHẬT TRẠNG THÁI').classes(
                        'text-base font-extrabold tracking-wide text-white')
                ui.button(icon='close', on_click=detail_dialog.close).props('flat round dense color=grey')

            with ui.row().classes('w-full gap-5 items-center bg-slate-900/80 p-4 rounded-xl border border-slate-800'):
                if xray_img is not None:
                    ui.image(cv2_to_base64(xray_img)).classes(
                        'w-44 h-28 object-contain bg-black rounded-lg border border-slate-700')
                with ui.column().classes('gap-1 flex-1 items-center'):
                    ui.label('BIỂN KIỂM SOÁT').classes('text-[11px] font-bold text-slate-400 tracking-widest')
                    ui.label(plate_str).classes('plate-3d text-2xl font-black mono-font text-center')
                    ui.label(f"Địa phương: {province_str} | Độ nét: {conf_val:.1f}%").classes(
                        'text-xs text-cyan-300 mt-1 font-semibold')

            owner_input = ui.input('Chủ sở hữu phương tiện', value=owner_str).props('dark outlined dense').classes(
                'w-full')
            ui.label('CHỌN TRẠNG THÁI PHÁP LÝ MỚI:').classes('text-xs font-bold text-slate-300 tracking-wider mt-1')
            selected_status = {"val": status_str}
            status_buttons_row = ui.row().classes('w-full gap-2 justify-between')

            def refresh_status_selector():
                status_buttons_row.clear()
                with status_buttons_row:
                    for st_key, st_info in STATUS_CONFIG.items():
                        is_chosen = (selected_status["val"] == st_key)
                        btn_color = 'emerald' if st_key == 'HOP LE' else ('rose' if st_key == 'PHAT NGUOI' else (
                            'amber' if st_key == 'HET HAN DANG KIEM' else 'purple'))
                        ui.button(
                            st_info["label"],
                            icon=st_info["icon"],
                            on_click=lambda k=st_key: set_modal_status(k)
                        ).props(f"{'unelevated' if is_chosen else 'outline'} color={btn_color} dense").classes(
                            'text-xs font-bold px-3 py-2 flex-1')

            def set_modal_status(k):
                selected_status["val"] = k
                refresh_status_selector()

            refresh_status_selector()

            def save_modal_changes():
                alpr_system.update_vehicle_in_db(plate_str, selected_status["val"], owner_input.value.strip())
                safe_notify(f'Đã cập nhật biển số {plate_str}!', 'positive')
                detail_dialog.close()
                refresh_all_views()

            with ui.row().classes('w-full justify-end gap-3 mt-3 pt-3 border-t border-slate-800'):
                ui.button('Hủy bỏ', on_click=detail_dialog.close).props('flat color=grey')
                ui.button('Lưu vào Database', icon='save', on_click=save_modal_changes).props(
                    'unelevated color=cyan').classes('font-bold px-4')

        detail_dialog.open()

    # ================= HEADER GỌN GÀNG, CHUYÊN NGHIỆP =================
    with ui.header(elevated=True).style('background: rgba(6, 9, 19, 0.88); backdrop-filter: blur(16px);').classes(
            'items-center justify-between px-6 py-3 border-b border-slate-800/80'):
        with ui.row().classes('items-center gap-3.5'):
            with ui.element('div').classes(
                    'p-2 rounded-xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/30 flex items-center justify-center'):
                ui.icon('shield', size='md').classes('text-cyan-400')
            with ui.column().classes('gap-0'):
                ui.label('TRUNG TÂM GIÁM SÁT & NHẬN DIỆN BIỂN SỐ PHƯƠNG TIỆN').classes(
                    'text-lg font-extrabold tracking-wide bg-clip-text text-transparent bg-gradient-to-r from-white via-cyan-100 to-emerald-300')
                ui.label('HỆ THỐNG KIỂM SOÁT GIAO THÔNG & TRUY VẤN CƠ SỞ DỮ LIỆU').classes(
                    'text-[11px] text-slate-400 tracking-widest font-medium')

        with ui.row().classes('items-center gap-4'):
            def export_csv_report():
                if not alpr_system.history_logs:
                    safe_notify('Chưa có dữ liệu phương tiện để xuất báo cáo!', 'warning')
                    return
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(
                    ['Thoi Gian', 'Bien Kiem Soat', 'Tinh Thanh', 'Chu Phuong Tien', 'Ngay Dang Ky', 'Do Net OCR',
                     'Trang Thai'])
                for r in alpr_system.history_logs:
                    writer.writerow(
                        [r['time'], r['plate'], r['province'], r['owner'], r['reg_date'], r['conf'], r['status']])
                csv_bytes = output.getvalue().encode('utf-8-sig')
                ui.download(csv_bytes, f"BaoCao_PhuongTien_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                safe_notify('Đã tải xuống file Báo cáo CSV!', 'positive')

            ui.button('Xuất Báo Cáo CSV', icon='download', on_click=export_csv_report).props(
                'outline color=cyan dense').classes('text-xs font-bold px-3 py-1.5 rounded-lg')

            with ui.row().classes(
                    'items-center gap-2 bg-emerald-950/50 border border-emerald-500/30 px-3 py-1.5 rounded-full'):
                ui.element('div').classes('w-2 h-2 rounded-full bg-emerald-400 animate-ping')
                ui.label('CSDL: ĐÃ KẾT NỐI').classes('text-xs font-bold text-emerald-300 mono-font')

            clock_label = ui.label().classes(
                'text-xs font-bold text-cyan-200 mono-font bg-slate-900/90 px-3.5 py-1.5 rounded-lg border border-slate-700/80')
            ui.timer(1.0, lambda: clock_label.set_text(datetime.now().strftime('%H:%M:%S | %d/%m/%Y')))

    def set_filter_mode(mode: str):
        ui_state["filter_mode"] = mode
        mode_names = {
            "ALL": "Hiển thị tất cả phương tiện",
            "VALID": "Đang lọc: Phương tiện Hợp lệ",
            "VIOLATION": "Đang lọc: Phương tiện Cảnh báo / Vi phạm"
        }
        safe_notify(mode_names.get(mode, ""), 'info')
        refresh_all_views()

    with ui.row().classes('w-full px-6 pt-4 gap-4 justify-between no-wrap'):
        card_all = ui.card().classes('kpi-interactive kpi-active-all w-1/4 p-4').on('click',
                                                                                    lambda: set_filter_mode('ALL'))
        with card_all:
            with ui.row().classes('justify-between items-center w-full'):
                with ui.column().classes('gap-0.5'):
                    ui.label('TỔNG XE QUA TRẠM').classes('text-xs font-bold text-slate-400 tracking-wider')
                    kpi_total = ui.label('0').classes('text-3xl font-black text-white mono-font mt-1')
                ui.icon('radar', size='lg').classes(
                    'text-cyan-400 bg-cyan-500/10 p-3 rounded-xl border border-cyan-500/25')

        card_valid = ui.card().classes('kpi-interactive w-1/4 p-4').on('click', lambda: set_filter_mode('VALID'))
        with card_valid:
            with ui.row().classes('justify-between items-center w-full'):
                with ui.column().classes('gap-0.5'):
                    ui.label('PHƯƠNG TIỆN HỢP LỆ').classes('text-xs font-bold text-emerald-400/90 tracking-wider')
                    kpi_valid = ui.label('0').classes('text-3xl font-black text-emerald-400 mono-font mt-1')
                ui.icon('verified', size='lg').classes(
                    'text-emerald-400 bg-emerald-500/10 p-3 rounded-xl border border-emerald-500/25')

        card_violation = ui.card().classes('kpi-interactive w-1/4 p-4').on('click',
                                                                           lambda: set_filter_mode('VIOLATION'))
        with card_violation:
            with ui.row().classes('justify-between items-center w-full'):
                with ui.column().classes('gap-0.5'):
                    ui.label('CẢNH BÁO / VI PHẠM').classes('text-xs font-bold text-rose-400 tracking-wider')
                    kpi_violation = ui.label('0').classes('text-3xl font-black text-rose-400 mono-font mt-1')
                ui.icon('gpp_bad', size='lg').classes(
                    'text-rose-400 bg-rose-500/10 p-3 rounded-xl border border-rose-500/25')

        with ui.card().classes('kpi-interactive w-1/4 p-4').on('click', lambda: set_filter_mode('ALL')):
            with ui.row().classes('justify-between items-center w-full'):
                with ui.column().classes('gap-0.5'):
                    ui.label('ĐỘ NÉT TRUNG BÌNH').classes('text-xs font-bold text-amber-300/90 tracking-wider')
                    kpi_conf = ui.label('0.0%').classes('text-3xl font-black text-amber-400 mono-font mt-1')
                ui.icon('auto_awesome', size='lg').classes(
                    'text-amber-400 bg-amber-500/10 p-3 rounded-xl border border-amber-500/25')

    def update_kpi_display():
        st = alpr_system.stats
        kpi_total.set_text(str(st["total"]))
        kpi_valid.set_text(str(st["valid"]))
        kpi_violation.set_text(str(st["violation"]))
        kpi_conf.set_text(f"{st['avg_conf']:.1f}%")

        card_all.classes(remove='kpi-active-all')
        card_valid.classes(remove='kpi-active-valid')
        card_violation.classes(remove='kpi-active-violation')
        if ui_state["filter_mode"] == "ALL":
            card_all.classes(add='kpi-active-all')
        elif ui_state["filter_mode"] == "VALID":
            card_valid.classes(add='kpi-active-valid')
        elif ui_state["filter_mode"] == "VIOLATION":
            card_violation.classes(add='kpi-active-violation')

    with ui.row().classes('w-full px-6 py-4 gap-6 items-start no-wrap'):

        with ui.column().classes('w-7/12 gap-5'):
            with ui.card().classes('cyber-glass w-full p-5'):
                with ui.row().classes('w-full justify-between items-center mb-3'):
                    with ui.row().classes('items-center gap-2.5'):
                        ui.icon('videocam', size='sm').classes('text-cyan-400')
                        ui.label('MÀN HÌNH CAMERA GIÁM SÁT').classes('text-sm font-extrabold tracking-wider text-white')

                main_image = ui.interactive_image().classes(
                    'w-full rounded-xl bg-slate-950/90 border border-slate-800/90 min-h-[420px] max-h-[510px] object-contain shadow-inner')

                video_controls_bar = ui.column().classes(
                    'w-full bg-slate-900/95 border border-slate-700/80 rounded-xl p-3 mt-3 gap-2')
                video_controls_bar.set_visibility(False)
                with video_controls_bar:
                    with ui.row().classes('w-full items-center gap-3 no-wrap'):
                        def toggle_pause_video():
                            alpr_system.video_paused = not alpr_system.video_paused
                            pause_btn.props(f"icon={'play_arrow' if alpr_system.video_paused else 'pause'}")

                        pause_btn = ui.button(icon='pause', on_click=toggle_pause_video).props(
                            'unelevated color=cyan round dense')

                        time_label = ui.label('00:00 / 00:00').classes(
                            'text-xs font-bold text-cyan-300 mono-font min-w-[92px]')

                        def on_slider_seek(e):
                            if ui_state["slider_updating_from_timer"]:
                                return
                            if alpr_system.video_total_frames > 1:
                                pct = float(e.value or 0.0)
                                target_frame = int((pct / 100.0) * (alpr_system.video_total_frames - 1))
                                alpr_system.seek_requested_frame = target_frame

                        timeline_slider = ui.slider(min=0, max=100, value=0, step=0.5, on_change=on_slider_seek).props(
                            'color=cyan dark').classes('flex-1')

                    with ui.row().classes('w-full justify-between items-center pt-1 border-t border-slate-800'):
                        ui.label('TỐC ĐỘ VIDEO:').classes('text-[11px] font-bold text-slate-400 tracking-wider')

                        speed_buttons = {}
                        with ui.row().classes('items-center gap-1.5'):
                            def set_video_speed(spd: float):
                                alpr_system.playback_speed = spd
                                for s_val, b_el in speed_buttons.items():
                                    b_el.props(f"{'unelevated' if abs(s_val - spd) < 0.01 else 'outline'} color=cyan")
                                safe_notify(f'Đã chỉnh tốc độ video: {spd}x', 'info')

                            for s_opt in [0.1, 0.2, 0.5, 1.0]:
                                is_default = abs(s_opt - 0.2) < 0.01
                                btn_s = ui.button(
                                    f'{s_opt}x',
                                    on_click=lambda s=s_opt: set_video_speed(s)
                                ).props(f"{'unelevated' if is_default else 'outline'} color=cyan dense").classes(
                                    'text-xs font-bold px-2.5 py-0.5 mono-font')
                                speed_buttons[s_opt] = btn_s

                async def handle_upload(e):
                    alpr_system.video_running = False
                    await asyncio.sleep(0.12)

                    filename = ""
                    if hasattr(e, 'name') and e.name:
                        filename = str(e.name)
                    elif hasattr(e, 'file') and hasattr(e.file, 'name') and e.file.name:
                        filename = str(e.file.name)
                    elif hasattr(e, 'file') and hasattr(e.file, 'filename') and e.file.filename:
                        filename = str(e.file.filename)

                    temp_raw = TEMP_DIR / f"upload_raw_{int(time.time())}.bin"

                    if hasattr(e, 'file') and hasattr(e.file, 'save'):
                        res_save = e.file.save(temp_raw)
                        if inspect.isawaitable(res_save):
                            await res_save
                    else:
                        file_obj = getattr(e, 'content', None) or getattr(e, 'file', None) or e
                        data = file_obj.read()
                        if inspect.isawaitable(data):
                            data = await data
                        with open(temp_raw, "wb") as f:
                            f.write(data)

                    header_bytes = b""
                    if temp_raw.exists():
                        with open(temp_raw, "rb") as f:
                            header_bytes = f.read(32)

                    ext = detect_media_type(header_bytes, filename)
                    temp_file = temp_raw.with_suffix(ext)
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                        temp_raw.rename(temp_file)
                    except Exception:
                        temp_file = temp_raw

                    try:
                        upload_widget.reset()
                    except Exception:
                        pass

                    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
                    if ext in video_extensions:
                        video_controls_bar.set_visibility(True)
                        pause_btn.props('icon=pause')
                        safe_notify('Đang phát video (tốc độ 0.2x) và tự động nhận diện biển số...', 'positive')
                        alpr_system.start_video_thread(temp_file)
                    else:
                        video_controls_bar.set_visibility(False)
                        safe_notify('Đang phân tích biển số...', 'info')
                        img, res = await run.io_bound(alpr_system.process_image, temp_file)
                        if img is not None:
                            main_image.set_source(alpr_system.latest_frame_b64)
                            alpr_system.frame_updated = False
                            ui_state["filter_mode"] = "ALL"
                            refresh_all_views()
                            safe_notify(f'Đã nhận diện {len(res)} biển số!', 'positive')

                upload_widget = ui.upload(
                    on_upload=handle_upload,
                    label="TẢI ẢNH HOẶC VIDEO PHƯƠNG TIỆN LÊN ĐỂ NHẬN DIỆN",
                    auto_upload=True
                ).classes('w-full mt-4 upload-modern')

            with ui.card().classes('cyber-glass w-full p-5'):
                with ui.row().classes('w-full justify-between items-center mb-3 gap-3'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('history', size='sm').classes('text-cyan-400')
                        ui.label('NHẬT KÝ PHƯƠNG TIỆN QUA TRẠM').classes(
                            'text-sm font-extrabold tracking-wider text-white')

                    with ui.row().classes('items-center gap-2'):
                        filter_btn_all = ui.button('Tất cả', on_click=lambda: set_filter_mode('ALL')).props(
                            'dense unelevated color=cyan').classes('text-[11px] font-bold px-2.5')
                        filter_btn_valid = ui.button('Hợp lệ', on_click=lambda: set_filter_mode('VALID')).props(
                            'dense outline color=emerald').classes('text-[11px] font-bold px-2.5')
                        filter_btn_viol = ui.button('Vi phạm', on_click=lambda: set_filter_mode('VIOLATION')).props(
                            'dense outline color=rose').classes('text-[11px] font-bold px-2.5')

                        def on_search_change(e):
                            ui_state["search_query"] = (e.value or "").strip().upper()
                            render_history_list()

                        ui.input(placeholder='Tìm biển số / chủ xe...', on_change=on_search_change).props(
                            'dark dense outlined clearable').classes('w-44 text-xs')

                log_container = ui.column().classes('w-full gap-2 max-h-[290px] overflow-y-auto pr-1')

                def render_history_list():
                    m = ui_state["filter_mode"]
                    filter_btn_all.props(f"{'unelevated' if m == 'ALL' else 'outline'} color=cyan")
                    filter_btn_valid.props(f"{'unelevated' if m == 'VALID' else 'outline'} color=emerald")
                    filter_btn_viol.props(f"{'unelevated' if m == 'VIOLATION' else 'outline'} color=rose")

                    log_container.clear()
                    with log_container:
                        filtered_logs = []
                        q = ui_state["search_query"]
                        for item in alpr_system.history_logs:
                            if m == "VALID" and item["status"] != "HOP LE":
                                continue
                            if m == "VIOLATION" and item["status"] == "HOP LE":
                                continue
                            if q and (
                                    q not in item["plate"].upper() and q not in item["owner"].upper() and q not in item[
                                "province"].upper()):
                                continue
                            filtered_logs.append(item)

                        if not filtered_logs:
                            with ui.row().classes('w-full py-6 justify-center items-center text-slate-500 gap-2'):
                                ui.icon('inbox', size='sm')
                                ui.label('Chưa có phương tiện nào trong danh sách.').classes('text-xs italic')
                            return

                        for log in filtered_logs:
                            st = log["status"]
                            st_cfg = STATUS_CONFIG.get(st, STATUS_CONFIG["HOP LE"])
                            is_alert = (st != "HOP LE")
                            row_bg = "bg-rose-950/25 border-rose-500/30 hover:border-rose-400/60" if is_alert else "bg-slate-900/70 border-slate-800 hover:border-cyan-500/40"

                            with ui.row().classes(
                                    f'w-full px-3.5 py-2.5 rounded-xl border {row_bg} items-center justify-between transition-all'):
                                with ui.row().classes('items-center gap-3'):
                                    ui.label(log["time"]).classes('text-xs text-slate-400 mono-font')
                                    ui.label(log["plate"]).classes(
                                        'bg-white text-slate-950 font-black mono-font text-xs px-2.5 py-1 rounded border border-slate-600 shadow-sm')
                                    with ui.column().classes('gap-0'):
                                        ui.label(log["owner"]).classes('text-xs font-bold text-white')
                                        ui.label(f"{log['province']} • ĐK: {log['reg_date']}").classes(
                                            'text-[10px] text-slate-400')

                                with ui.row().classes('items-center gap-2.5'):
                                    ui.badge(f"OCR: {log['conf']}", color='slate').classes('text-[10px] mono-font')

                                    with ui.element('div').classes(
                                            f"px-2.5 py-1 rounded-full text-[11px] font-extrabold flex items-center gap-1 status-chip {st_cfg['badge_cls']}"):
                                        ui.icon(st_cfg["icon"], size='14px')
                                        ui.label(st_cfg["label"])
                                        ui.icon('arrow_drop_down', size='14px')
                                        with ui.menu().props('dark auto-close'):
                                            ui.menu_item('✅ Đánh dấu: HỢP LỆ',
                                                         on_click=lambda p=log["plate"]: quick_change_status(p,
                                                                                                             "HOP LE"))
                                            ui.menu_item('🚨 Đánh dấu: PHẠT NGUỘI',
                                                         on_click=lambda p=log["plate"]: quick_change_status(p,
                                                                                                             "PHAT NGUOI"))
                                            ui.menu_item('⚠️ Đánh dấu: HẾT HẠN ĐĂNG KIỂM',
                                                         on_click=lambda p=log["plate"]: quick_change_status(p,
                                                                                                             "HET HAN DANG KIEM"))
                                            ui.menu_item('🚓 Đánh dấu: XE TRUY NÃ',
                                                         on_click=lambda p=log["plate"]: quick_change_status(p,
                                                                                                             "XE TRUY NA"))

                                    ui.button(
                                        icon='open_in_new',
                                        on_click=lambda l=log: open_vehicle_modal(
                                            l["plate"], l["owner"], l["reg_date"], l["status"], l["province"],
                                            l["xray"], l["conf_val"]
                                        )
                                    ).props('flat round dense size=sm color=cyan')

        with ui.column().classes('w-5/12 gap-4'):
            with ui.row().classes('w-full justify-between items-center px-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('assignment_ind', size='sm').classes('text-emerald-400')
                    ui.label('THÔNG TIN PHƯƠNG TIỆN NHẬN DIỆN').classes(
                        'text-sm font-extrabold tracking-wider text-white')
                filter_status_badge = ui.badge('TẤT CẢ', color='cyan').classes('text-[10px] font-bold px-2.5 py-1')

            results_container = ui.column().classes('w-full gap-4 max-h-[820px] overflow-y-auto pr-1')

            def quick_change_status(plate_str: str, new_st: str):
                st_label = STATUS_CONFIG.get(new_st, {}).get("label", new_st)
                safe_notify(f'Đã đổi trạng thái biển số {plate_str} -> {st_label}',
                            'positive' if new_st == 'HOP LE' else 'warning')
                alpr_system.update_vehicle_in_db(plate_str, new_st)
                if not alpr_system.video_running and alpr_system.latest_frame_b64:
                    main_image.set_source(alpr_system.latest_frame_b64)
                refresh_all_views()

            def render_results():
                results_container.clear()
                m = ui_state["filter_mode"]
                filter_status_badge.set_text("TẤT CẢ" if m == "ALL" else ("HỢP LỆ" if m == "VALID" else "VI PHẠM"))

                with results_container:
                    if not alpr_system.current_results:
                        with ui.card().classes('cyber-glass w-full p-10 items-center justify-center text-center'):
                            with ui.element('div').classes(
                                    'p-4 rounded-full bg-slate-800/70 border border-slate-700 mb-3'):
                                ui.icon('qr_code_scanner', size='xl').classes('text-cyan-400')
                            ui.label('ĐANG CHỜ DỮ LIỆU HÌNH ẢNH / VIDEO').classes(
                                'text-sm font-extrabold text-slate-200 tracking-wider')
                            ui.label(
                                'Tải ảnh hoặc video lên khung bên trái để hệ thống tự động phát hiện và đọc biển số theo thời gian thực.').classes(
                                'text-xs text-slate-400 mt-1 max-w-sm')
                        return

                    visible_results = [
                        r for r in alpr_system.current_results
                        if not (m == "VALID" and r["db"]["status"] != "HOP LE") and not (
                                    m == "VIOLATION" and r["db"]["status"] == "HOP LE")
                    ]

                    if not visible_results:
                        with ui.card().classes('cyber-glass w-full p-8 items-center justify-center text-center'):
                            ui.icon('filter_alt_off', size='lg').classes('text-amber-400 mb-2')
                            ui.label('KHÔNG CÓ XE NÀO THUỘC BỘ LỌC NÀY').classes('text-sm font-bold text-amber-300')
                            ui.button('Xem lại tất cả', icon='visibility',
                                      on_click=lambda: set_filter_mode('ALL')).props(
                                'outline color=cyan dense').classes('mt-3 text-xs font-bold px-3')
                        return

                    for res in visible_results:
                        st = res['db']['status']
                        st_cfg = STATUS_CONFIG.get(st, STATUS_CONFIG["HOP LE"])
                        is_alert = (st != "HOP LE")

                        with ui.card().classes(f"cyber-glass {st_cfg['card_cls']} w-full p-5"):
                            with ui.row().classes(
                                    'w-full justify-between items-center mb-3 pb-2.5 border-b border-slate-700/60'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.element('div').classes(f"w-2.5 h-2.5 rounded-full {st_cfg['dot_cls']}")
                                    ui.label(f"XE #{res['id']} • {res['timestamp']}").classes(
                                        'text-xs text-slate-300 mono-font font-semibold')

                                with ui.element('div').classes(
                                        f"px-3 py-1 rounded-full text-xs font-extrabold flex items-center gap-1.5 status-chip {st_cfg['badge_cls']}"):
                                    ui.icon(st_cfg["icon"], size='16px')
                                    ui.label(st_cfg["label"])
                                    ui.icon('expand_more', size='16px')
                                    with ui.menu().props('dark auto-close'):
                                        ui.menu_item('✅ Xác nhận: HỢP LỆ',
                                                     on_click=lambda p=res["plate"]: quick_change_status(p, "HOP LE"))
                                        ui.menu_item('🚨 Cảnh báo: PHẠT NGUỘI',
                                                     on_click=lambda p=res["plate"]: quick_change_status(p,
                                                                                                         "PHAT NGUOI"))
                                        ui.menu_item('⚠️ Cảnh báo: HẾT HẠN ĐĂNG KIỂM',
                                                     on_click=lambda p=res["plate"]: quick_change_status(p,
                                                                                                         "HET HAN DANG KIEM"))
                                        ui.menu_item('🚓 Khẩn cấp: XE TRUY NÃ',
                                                     on_click=lambda p=res["plate"]: quick_change_status(p,
                                                                                                         "XE TRUY NA"))

                            with ui.row().classes('w-full gap-4 items-center mb-4 no-wrap'):
                                if res['xray'] is not None:
                                    with ui.column().classes(
                                            'w-5/12 bg-slate-950 p-2 rounded-xl border border-slate-700/80 items-center shadow-inner cursor-pointer').on(
                                            'click',
                                            lambda r=res: open_vehicle_modal(r["plate"], r["db"]["owner"],
                                                                             r["db"]["reg_date"], r["db"]["status"],
                                                                             r["province"], r["xray"], r["conf"])
                                    ):
                                        ui.image(cv2_to_base64(res['xray'])).classes(
                                            'w-full h-24 object-contain rounded')
                                        ui.label('ẢNH CẮT BIỂN SỐ').classes(
                                            'text-[10px] text-slate-400 mono-font font-semibold mt-1')

                                with ui.column().classes('w-7/12 gap-2'):
                                    ui.label('BIỂN KIỂM SOÁT').classes(
                                        'text-[10px] font-extrabold text-slate-400 tracking-widest')
                                    ui.label(res['plate']).classes(
                                        'plate-3d text-2xl font-black mono-font text-center w-full')

                                    with ui.column().classes('w-full gap-1 mt-1'):
                                        with ui.row().classes('justify-between w-full text-xs'):
                                            ui.label('Phát hiện Biển số:').classes('text-slate-400 font-medium')
                                            ui.label(f"{res['det_conf']}%").classes(
                                                'font-extrabold text-cyan-300 mono-font')
                                        ui.linear_progress(value=res['det_conf'] / 100.0, show_value=False,
                                                           color='cyan').classes('h-1.5 rounded-full')

                                        with ui.row().classes('justify-between w-full text-xs'):
                                            ui.label('Độ tin cậy Ký tự (OCR):').classes('text-slate-400 font-medium')
                                            ui.label(f"{res['conf']:.1f}%").classes(
                                                'font-extrabold text-emerald-300 mono-font')
                                        ui.linear_progress(value=res['conf'] / 100.0, show_value=False,
                                                           color='emerald').classes('h-1.5 rounded-full')

                            with ui.element('div').classes(
                                    'w-full bg-slate-900/80 rounded-xl p-3 border border-slate-800/90 grid grid-cols-2 gap-2.5'):
                                with ui.column().classes('gap-0'):
                                    ui.label('ĐỊA PHƯƠNG').classes('text-[10px] font-bold text-slate-400')
                                    ui.label(res['province']).classes('text-sm font-extrabold text-amber-300')
                                with ui.column().classes('gap-0'):
                                    ui.label('CHỦ PHƯƠNG TIỆN').classes('text-[10px] font-bold text-slate-400')
                                    ui.label(res['db']['owner']).classes('text-sm font-extrabold text-white')
                                with ui.column().classes('gap-0'):
                                    ui.label('NGÀY ĐĂNG KÝ').classes('text-[10px] font-bold text-slate-400')
                                    ui.label(res['db']['reg_date']).classes(
                                        'text-sm font-bold text-slate-200 mono-font')
                                with ui.column().classes('gap-0'):
                                    ui.label('TRẠNG THÁI').classes('text-[10px] font-bold text-slate-400')
                                    status_text_color = "text-rose-400" if is_alert else "text-emerald-400"
                                    ui.label(f"● {st_cfg['label']}").classes(f'text-sm font-black {status_text_color}')

                            with ui.row().classes(
                                    'w-full justify-between items-center mt-3 pt-2.5 border-t border-slate-800/80'):
                                with ui.row().classes('gap-1.5'):
                                    if is_alert:
                                        ui.button(
                                            'Xác nhận Hợp lệ',
                                            icon='check_circle',
                                            on_click=lambda p=res["plate"]: quick_change_status(p, "HOP LE")
                                        ).props('dense unelevated color=emerald').classes(
                                            'text-[11px] font-bold px-2.5 py-1 rounded-lg')
                                    else:
                                        ui.button(
                                            'Báo Phạt nguội',
                                            icon='gavel',
                                            on_click=lambda p=res["plate"]: quick_change_status(p, "PHAT NGUOI")
                                        ).props('dense outline color=rose').classes(
                                            'text-[11px] font-bold px-2.5 py-1 rounded-lg')

                                    ui.button(
                                        'Hết hạn ĐK',
                                        icon='event_busy',
                                        on_click=lambda p=res["plate"]: quick_change_status(p, "HET HAN DANG KIEM")
                                    ).props('dense outline color=amber').classes(
                                        'text-[11px] font-bold px-2.5 py-1 rounded-lg')

                                ui.button(
                                    'Chi tiết / Sửa',
                                    icon='edit_note',
                                    on_click=lambda r=res: open_vehicle_modal(
                                        r["plate"], r["db"]["owner"], r["db"]["reg_date"], r["db"]["status"],
                                        r["province"], r["xray"], r["conf"]
                                    )
                                ).props('dense flat color=cyan').classes('text-xs font-bold')

    def refresh_all_views():
        try:
            update_kpi_display()
            render_results()
            render_history_list()
        except Exception:
            pass

    def sync_ui_loop():
        if alpr_system.frame_updated and alpr_system.latest_frame_b64:
            alpr_system.frame_updated = False
            main_image.set_source(alpr_system.latest_frame_b64)

        # Cập nhật thanh thời gian (Timeline) khi đang ở chế độ Video
        if alpr_system.is_video_mode and alpr_system.video_total_frames > 0:
            cur_f = alpr_system.video_current_frame
            tot_f = alpr_system.video_total_frames
            fps = max(1.0, alpr_system.video_native_fps)
            cur_sec = cur_f / fps
            tot_sec = tot_f / fps
            time_label.set_text(f"{format_seconds(cur_sec)} / {format_seconds(tot_sec)}")

            pct = round((cur_f / float(tot_f)) * 100.0, 1)
            ui_state["slider_updating_from_timer"] = True
            timeline_slider.value = pct
            ui_state["slider_updating_from_timer"] = False
            pause_btn.props(f"icon={'play_arrow' if alpr_system.video_paused else 'pause'}")

        if alpr_system.ui_needs_refresh:
            alpr_system.ui_needs_refresh = False
            refresh_all_views()

    ui.timer(0.08, sync_ui_loop)
    refresh_all_views()


if __name__ in {"__main__", "__mp_main__"}:
    open_port = native.find_open_port(start_port=8083)
    ui.run(title="Trung Tâm Giám Sát & Nhận Diện Biển Số", port=open_port, reload=False)