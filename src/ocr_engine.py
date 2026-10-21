import cv2
import numpy as np
from ultralytics import YOLO
from src.config import MODELS_DIR, OCR_MODEL_NAME, VALID_PROVINCE_CODES, setup_logger
from src.preprocessing import PlatePreprocessor

logger = setup_logger("OCREngine")


class GeometrySorter:

    LETTER_TO_DIGIT = {
        'O': '0', 'D': '0', 'Q': '0', 'U': '0', 'C': '0',
        'I': '1', 'L': '1', 'T': '1', 'J': '1',
        'Z': '2',
        'S': '5', 'F': '5',
        'G': '6', 'E': '6',
        'B': '8',
        'A': '4', 'H': '4', 'M': '4'
    }
    DIGIT_TO_LETTER = {
        '0': 'D',
        '1': 'A',
        '2': 'Z',
        '3': 'C',
        '4': 'A',
        '5': 'S',
        '6': 'C',
        '7': 'F',
        '8': 'B'
    }

    @staticmethod
    def _get_binary_char_crop(char_item: dict, plate_img: np.ndarray):
        if plate_img is None or plate_img.size == 0:
            return None
        h_img, w_img = plate_img.shape[:2]
        x1, y1 = max(0, int(char_item['x1'])), max(0, int(char_item['y1']))
        x2, y2 = min(w_img, int(char_item['x2'])), min(h_img, int(char_item['y2']))

        crop = plate_img[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 5:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @classmethod
    def verify_c_vs_e(cls, char_item: dict, plate_img: np.ndarray) -> str:
        label = char_item['label']
        if label not in ['E', 'G'] or plate_img is None:
            return label

        binary = cls._get_binary_char_crop(char_item, plate_img)
        if binary is None:
            return label

        ch, cw = binary.shape[:2]
        mid_right = binary[int(ch * 0.36):int(ch * 0.64), int(cw * 0.45):int(cw * 0.90)]
        top_bar = binary[int(ch * 0.08):int(ch * 0.28), int(cw * 0.30):int(cw * 0.80)]
        bot_bar = binary[int(ch * 0.72):int(ch * 0.92), int(cw * 0.30):int(cw * 0.80)]

        if mid_right.size > 0 and top_bar.size > 0 and bot_bar.size > 0:
            white_mid = np.sum(mid_right == 255) / float(mid_right.size)
            black_top = np.sum(top_bar == 0) / float(top_bar.size)
            black_bot = np.sum(bot_bar == 0) / float(bot_bar.size)

            if white_mid > 0.52 and (black_top > 0.25 or black_bot > 0.25):
                return 'C'
        return label

    @classmethod
    def verify_x_vs_k(cls, char_item: dict, plate_img: np.ndarray) -> str:
        label = char_item['label']
        if label != 'K' or plate_img is None:
            return label

        binary = cls._get_binary_char_crop(char_item, plate_img)
        if binary is None:
            return label

        ch, cw = binary.shape[:2]
        left_mid_gap = binary[int(ch * 0.35):int(ch * 0.65), int(cw * 0.05):int(cw * 0.35)]
        center_cross = binary[int(ch * 0.38):int(ch * 0.62), int(cw * 0.35):int(cw * 0.65)]

        if left_mid_gap.size > 0 and center_cross.size > 0:
            white_left_mid = np.sum(left_mid_gap == 255) / float(left_mid_gap.size)
            black_center = np.sum(center_cross == 0) / float(center_cross.size)

            if white_left_mid > 0.55 and black_center > 0.40:
                return 'X'
        return label

    @classmethod
    def verify_5_6_8(cls, char_item: dict, plate_img: np.ndarray) -> str:
        label = char_item['label']
        if label not in ['5', '6', '8'] or plate_img is None:
            return label

        binary = cls._get_binary_char_crop(char_item, plate_img)
        if binary is None:
            return label

        ch, cw = binary.shape[:2]
        top_right = binary[int(ch * 0.18):int(ch * 0.42), int(cw * 0.60):int(cw * 0.92)]
        bot_left = binary[int(ch * 0.55):int(ch * 0.80), int(cw * 0.08):int(cw * 0.40)]
        top_bar = binary[int(ch * 0.05):int(ch * 0.22), int(cw * 0.25):int(cw * 0.85)]

        if top_right.size > 0 and bot_left.size > 0 and top_bar.size > 0:
            tr_white = np.sum(top_right == 255) / float(top_right.size)
            bl_white = np.sum(bot_left == 255) / float(bot_left.size)
            top_black = np.sum(top_bar == 0) / float(top_bar.size)

            if label == '8' and tr_white > 0.56:
                return '5' if bl_white > 0.55 else '6'

            if label == '6' and bl_white > 0.54 and top_black > 0.42:
                return '5'

        return label

    @staticmethod
    def filter_boxes_and_border_noise(char_items: list, img_w: int, img_h: int, iou_thresh=0.45) -> list:
        if not char_items:
            return []

        valid_candidates = []
        for item in char_items:
            w, h = item['w'], item['h']
            cx, cy = item['cx'], item['cy']
            aspect = w / float(h) if h > 0 else 0

            if h < img_h * 0.14 or h > img_h * 0.92:
                continue
            if aspect > 1.38 or aspect < 0.07:
                continue
            if (cx < img_w * 0.02 or cx > img_w * 0.98) and aspect < 0.20:
                continue

            valid_candidates.append(item)

        if not valid_candidates:
            return []

        median_h = float(np.median([x['h'] for x in valid_candidates]))
        size_filtered = [
            x for x in valid_candidates
            if (0.48 * median_h) <= x['h'] <= (1.62 * median_h)
        ]
        if not size_filtered:
            size_filtered = valid_candidates

        sorted_items = sorted(size_filtered, key=lambda x: x['conf'], reverse=True)
        kept = []
        for item in sorted_items:
            is_duplicate = False
            for k in kept:
                inter_x1 = max(item['x1'], k['x1'])
                inter_y1 = max(item['y1'], k['y1'])
                inter_x2 = min(item['x2'], k['x2'])
                inter_y2 = min(item['y2'], k['y2'])

                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                min_area = min(item['w'] * item['h'], k['w'] * k['h'])

                dist_cx = abs(item['cx'] - k['cx'])
                dist_cy = abs(item['cy'] - k['cy'])
                avg_w = (item['w'] + k['w']) / 2.0
                avg_h = (item['h'] + k['h']) / 2.0

                if (min_area > 0 and (inter_area / min_area) > iou_thresh) or (dist_cx < avg_w * 0.38 and dist_cy < avg_h * 0.42):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(item)

        return kept

    @staticmethod
    def is_really_two_lines(items: list) -> bool:
        if len(items) < 4:
            return False

        avg_h = sum([x['h'] for x in items]) / len(items)
        min_cy = min([x['cy'] for x in items])
        max_cy = max([x['cy'] for x in items])
        y_spread = max_cy - min_cy

        if y_spread < 0.60 * avg_h:
            return False

        sorted_by_x = sorted(items, key=lambda x: x['cx'])
        x_overlaps = 0
        for i in range(len(sorted_by_x)):
            for j in range(i + 1, len(sorted_by_x)):
                avg_w = (sorted_by_x[i]['w'] + sorted_by_x[j]['w']) / 2.0
                if abs(sorted_by_x[i]['cx'] - sorted_by_x[j]['cx']) < avg_w * 0.65:
                    if abs(sorted_by_x[i]['cy'] - sorted_by_x[j]['cy']) > avg_h * 0.52:
                        x_overlaps += 1

        return x_overlaps >= 1

    @staticmethod
    def split_two_lines_sloped(items: list) -> tuple:
        ys = np.array([item['cy'] for item in items], dtype=np.float32)
        c1, c2 = float(np.min(ys)), float(np.max(ys))

        for _ in range(10):
            g1 = [y for y in ys if abs(y - c1) <= abs(y - c2)]
            g2 = [y for y in ys if abs(y - c1) > abs(y - c2)]
            if g1: c1 = float(np.mean(g1))
            if g2: c2 = float(np.mean(g2))

        top_init = [item for item in items if abs(item['cy'] - c1) <= abs(item['cy'] - c2)]
        bot_init = [item for item in items if abs(item['cy'] - c1) > abs(item['cy'] - c2)]

        ref_line = bot_init if len(bot_init) >= len(top_init) else top_init
        slope = 0.0
        if len(ref_line) >= 2:
            ref_line.sort(key=lambda x: x['cx'])
            dx = ref_line[-1]['cx'] - ref_line[0]['cx']
            dy = ref_line[-1]['cy'] - ref_line[0]['cy']
            if abs(dx) > 15:
                slope = float(np.clip(dy / dx, -0.45, 0.45))

        mean_cx = float(np.mean([item['cx'] for item in items]))
        corrected_ys = [item['cy'] - slope * (item['cx'] - mean_cx) for item in items]
        mid_y = (min(corrected_ys) + max(corrected_ys)) / 2.0

        top_line = [item for item, cy_c in zip(items, corrected_ys) if cy_c <= mid_y]
        bottom_line = [item for item, cy_c in zip(items, corrected_ys) if cy_c > mid_y]

        top_line.sort(key=lambda x: x['cx'])
        bottom_line.sort(key=lambda x: x['cx'])
        return top_line, bottom_line

    @classmethod
    def _apply_vn_rules(cls, top_items: list, bottom_items: list, is_two_lines: bool, plate_img: np.ndarray = None):
        if is_two_lines:
            for i, item in enumerate(top_items):
                c = item['label']
                if i in [0, 1]:
                    c = cls.LETTER_TO_DIGIT.get(c, c)
                    item['label'] = cls.verify_5_6_8({**item, 'label': c}, plate_img)
                elif i == 2:
                    c_fixed = cls.DIGIT_TO_LETTER.get(c, c)
                    c_fixed = cls.verify_c_vs_e({**item, 'label': c_fixed}, plate_img)
                    c_fixed = cls.verify_x_vs_k({**item, 'label': c_fixed}, plate_img)
                    item['label'] = c_fixed
                elif i == 3:
                    if c in ['O', 'Q', 'I', 'L', 'Z', 'S', 'B', 'G']:
                        item['label'] = cls.LETTER_TO_DIGIT.get(c, c)

            for item in bottom_items:
                c = cls.LETTER_TO_DIGIT.get(item['label'], item['label'])
                item['label'] = cls.verify_5_6_8({**item, 'label': c}, plate_img)

            return "".join([x['label'] for x in top_items]), "".join([x['label'] for x in bottom_items])
        else:
            total_len = len(top_items)
            for i, item in enumerate(top_items):
                c = item['label']
                if i in [0, 1]:
                    c = cls.LETTER_TO_DIGIT.get(c, c)
                    item['label'] = cls.verify_5_6_8({**item, 'label': c}, plate_img)
                elif i == 2:
                    c_fixed = cls.DIGIT_TO_LETTER.get(c, c)
                    c_fixed = cls.verify_c_vs_e({**item, 'label': c_fixed}, plate_img)
                    c_fixed = cls.verify_x_vs_k({**item, 'label': c_fixed}, plate_img)
                    item['label'] = c_fixed
                elif i == 3:
                    if total_len <= 8:
                        c = cls.LETTER_TO_DIGIT.get(c, c)
                        item['label'] = cls.verify_5_6_8({**item, 'label': c}, plate_img)
                elif i >= 4:
                    c = cls.LETTER_TO_DIGIT.get(c, c)
                    item['label'] = cls.verify_5_6_8({**item, 'label': c}, plate_img)

            return "".join([x['label'] for x in top_items]), ""

    @classmethod
    def sort_and_format(cls, char_items, img_w, img_h, plate_img=None):
        if not char_items:
            return "", [], 0, 0

        filtered = cls.filter_boxes_and_border_noise(char_items, img_w, img_h)
        if not filtered:
            return "", [], 0, 0

        is_two_lines = cls.is_really_two_lines(filtered)

        if not is_two_lines:
            filtered.sort(key=lambda x: x['cx'])
            if len(filtered) > 9:
                filtered = sorted(filtered, key=lambda x: x['conf'], reverse=True)[:9]
                filtered.sort(key=lambda x: x['cx'])

            clean_top, _ = cls._apply_vn_rules(filtered, [], is_two_lines=False, plate_img=plate_img)
            if len(clean_top) >= 7:
                province = clean_top[:2]
                remainder = clean_top[2:]
                num_len = 5 if len(clean_top) >= 8 else 4
                series = remainder[:-num_len]
                num_seq = remainder[-num_len:]
                num_fmt = f"{num_seq[:3]}.{num_seq[3:]}" if len(num_seq) == 5 else num_seq
                return f"{province}{series}-{num_fmt}", filtered, len(filtered), 0
            return clean_top, filtered, len(filtered), 0
        else:
            top_line, bottom_line = cls.split_two_lines_sloped(filtered)
            if len(top_line) > 4:
                top_line = sorted(top_line, key=lambda x: x['conf'], reverse=True)[:4]
                top_line.sort(key=lambda x: x['cx'])
            if len(bottom_line) > 5:
                bottom_line = sorted(bottom_line, key=lambda x: x['conf'], reverse=True)[:5]
                bottom_line.sort(key=lambda x: x['cx'])

            clean_top, clean_bot = cls._apply_vn_rules(top_line, bottom_line, is_two_lines=True, plate_img=plate_img)
            bot_fmt = f"{clean_bot[:3]}.{clean_bot[3:]}" if len(clean_bot) == 5 else clean_bot

            if len(clean_top) == 4:
                return f"{clean_top[:2]}-{clean_top[2:]} {bot_fmt}", top_line + bottom_line, len(top_line), len(bottom_line)
            elif len(clean_top) == 3:
                return f"{clean_top} {bot_fmt}", top_line + bottom_line, len(top_line), len(bottom_line)
            return f"{clean_top} {bot_fmt}".strip(), top_line + bottom_line, len(top_line), len(bottom_line)


class OCREngine:
    def __init__(self):
        logger.info("=== NẠP LÕI YOLO OCR (CHUYÊN TRỊ BIỂN VÀNG, MỜ, XA, NGHIÊNG PRO) ===")
        ocr_model_path = MODELS_DIR / OCR_MODEL_NAME
        self.model = YOLO(str(ocr_model_path)) if ocr_model_path.exists() else None
        self.class_names = self.model.names if self.model else []
        self.sorter = GeometrySorter()
        self.preprocessor = PlatePreprocessor()

    def _run_single_pass(self, variant_name: str, img_variant: np.ndarray, imgsz: int = 416):
        img_h, img_w = img_variant.shape[:2]
        results = self.model(img_variant, imgsz=imgsz, verbose=False, conf=0.09, iou=0.45)

        raw_items = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = str(self.class_names[cls_id]).upper()

                raw_items.append({
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                    'cx': float((x1 + x2) / 2.0), 'cy': float((y1 + y2) / 2.0),
                    'w': float(x2 - x1), 'h': float(y2 - y1),
                    'conf': conf, 'label': label
                })

        plate_text, valid_items, top_cnt, bot_cnt = self.sorter.sort_and_format(
            raw_items, img_w, img_h, plate_img=img_variant
        )
        avg_conf = (sum([x['conf'] for x in valid_items]) / len(valid_items) * 100.0) if valid_items else 0.0
        sum_conf = sum([x['conf'] for x in valid_items]) * 100.0

        char_count = len(valid_items)
        score = avg_conf * 1.5 + sum_conf * 0.45

        if bot_cnt > 0:
            if top_cnt in [3, 4] and bot_cnt == 5:
                score += 560.0
            elif top_cnt in [3, 4] and bot_cnt == 4:
                score += 390.0
            else:
                score -= 200.0
        else:
            if char_count in [8, 9]:
                score += 520.0
            elif char_count == 7:
                score += 350.0
            else:
                score -= 250.0

        clean_alnum = "".join([c for c in plate_text if c.isalnum()])
        if len(clean_alnum) >= 2:
            if clean_alnum[:2] in VALID_PROVINCE_CODES or clean_alnum[:2] == "80":
                score += 180.0
            else:
                score -= 220.0

        if len(clean_alnum) >= 3 and clean_alnum[2].isalpha():
            score += 80.0

        if "YellowToWhite_Priority" in variant_name:
            score += 45.0
        elif "Natural" in variant_name or "RedGreenClean" in variant_name:
            score += 20.0

        return {
            'score': score,
            'text': plate_text,
            'items': valid_items,
            'avg_conf': avg_conf,
            'img': img_variant,
            'name': f"{variant_name}_{imgsz}"
        }

    def extract_text(self, image: np.ndarray) -> tuple:
        if image is None or image.size == 0 or self.model is None:
            return "", image, 0.0

        variants = self.preprocessor.generate_variants(image)
        pass_results = []

        for v_name, v_img in variants:
            pass_results.append(self._run_single_pass(v_name, v_img, imgsz=320))
            pass_results.append(self._run_single_pass(v_name, v_img, imgsz=480))

        best_res = max(pass_results, key=lambda x: x['score'])

        debug_img = best_res['img'].copy()
        for item in best_res['items']:
            x1, y1, x2, y2 = int(item['x1']), int(item['y1']), int(item['x2']), int(item['y2'])
            char_label = item['label']
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_img, char_label, (x1 + 2, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA
            )

        return best_res['text'], debug_img, best_res['avg_conf']