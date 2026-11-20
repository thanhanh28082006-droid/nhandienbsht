import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from src.config import MODELS_DIR, OCR_MODEL_NAME, VALID_PROVINCE_CODES, setup_logger
from src.preprocessing import PlatePreprocessor

logger = setup_logger("OCREngine")


class GeometrySorter:
    """
    Sắp xếp tọa độ ký tự trên biển số 1 dòng / 2 dòng và áp dụng quy chuẩn biển số Việt Nam:
    - Tự động loại bỏ nhiễu mép trái trên biển 1 dòng (trị lỗi 129C-640.38 -> 29C-640.38).
    - Ép chuẩn chữ số ở các vị trí bắt buộc là số (trị lỗi 29C-640.3M -> 29C-640.38).
    """

    LETTER_TO_DIGIT_STRICT = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'L': '1', 'T': '1',
        'Z': '2',
        'A': '4', 'H': '4',
        'S': '5',
        'G': '6', 'E': '6',
        'B': '8', 'M': '8'
    }

    DIGIT_TO_LETTER_STRICT = {
        '0': 'D',
        '2': 'Z',
        '4': 'A',
        '5': 'S',
        '8': 'B'
    }

    @staticmethod
    def filter_boxes_and_border_noise(char_items: list, img_w: int, img_h: int, iou_thresh=0.45) -> list:
        if not char_items:
            return []

        valid_candidates = []
        for item in char_items:
            w, h = item['w'], item['h']
            cx, cy = item['cx'], item['cy']
            aspect = w / float(h) if h > 0 else 0

            if h < img_h * 0.12 or h > img_h * 0.94:
                continue
            if aspect > 1.45 or aspect < 0.04:
                continue
            if (cx < img_w * 0.012 or cx > img_w * 0.988) and aspect < 0.10:
                continue

            valid_candidates.append(item)

        if not valid_candidates:
            return []

        median_h = float(np.median([x['h'] for x in valid_candidates]))
        size_filtered = [
            x for x in valid_candidates
            if (0.45 * median_h) <= x['h'] <= (1.68 * median_h)
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

                if (min_area > 0 and (inter_area / min_area) > iou_thresh) or (dist_cx < avg_w * 0.36 and dist_cy < avg_h * 0.42):
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
        sorted_by_x = sorted(items, key=lambda x: x['cx'])
        dx = sorted_by_x[-1]['cx'] - sorted_by_x[0]['cx']
        dy = sorted_by_x[-1]['cy'] - sorted_by_x[0]['cy']
        slope = float(np.clip(dy / dx, -0.75, 0.75)) if abs(dx) > 20 else 0.0
        mean_cx = float(np.mean([x['cx'] for x in items]))

        corr_ys = [x['cy'] - slope * (x['cx'] - mean_cx) for x in items]
        y_spread = max(corr_ys) - min(corr_ys)

        if y_spread < 0.60 * avg_h:
            return False

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
            if g1:
                c1 = float(np.mean(g1))
            if g2:
                c2 = float(np.mean(g2))

        top_init = [item for item in items if abs(item['cy'] - c1) <= abs(item['cy'] - c2)]
        bot_init = [item for item in items if abs(item['cy'] - c1) > abs(item['cy'] - c2)]

        ref_line = bot_init if len(bot_init) >= len(top_init) else top_init
        slope = 0.0
        if len(ref_line) >= 2:
            ref_line.sort(key=lambda x: x['cx'])
            dx = ref_line[-1]['cx'] - ref_line[0]['cx']
            dy = ref_line[-1]['cy'] - ref_line[0]['cy']
            if abs(dx) > 15:
                slope = float(np.clip(dy / dx, -0.65, 0.65))

        mean_cx = float(np.mean([item['cx'] for item in items]))
        corrected_ys = [item['cy'] - slope * (item['cx'] - mean_cx) for item in items]
        mid_y = (min(corrected_ys) + max(corrected_ys)) / 2.0

        top_line = [item for item, cy_c in zip(items, corrected_ys) if cy_c <= mid_y]
        bottom_line = [item for item, cy_c in zip(items, corrected_ys) if cy_c > mid_y]

        top_line.sort(key=lambda x: x['cx'])
        bottom_line.sort(key=lambda x: x['cx'])
        return top_line, bottom_line

    @classmethod
    def _clean_one_line_leading_noise(cls, items: list) -> list:
        """
        Trên biển 1 dòng Việt Nam, vị trí 0-1 là 2 chữ số mã tỉnh, vị trí 2 là chữ cái series.
        Nếu phát hiện 9 ký tự mà 3 ký tự đầu đều là số và ký tự thứ 4 là chữ cái (VD: 129C64038),
        thì ký tự đầu tiên ('1') là vệt bóng/khung viền mép trái -> Loại bỏ để trả về 29C-640.38!
        """
        if len(items) == 9:
            c0, c1, c2, c3 = items[0]['label'], items[1]['label'], items[2]['label'], items[3]['label']
            if c0.isdigit() and c1.isdigit() and c2.isdigit() and c3.isalpha():
                return items[1:]
        return items

    @classmethod
    def _apply_vn_rules(cls, top_items: list, bottom_items: list, is_two_lines: bool):
        if is_two_lines:
            for i, item in enumerate(top_items):
                c = item['label']
                if i in [0, 1]:
                    item['label'] = cls.LETTER_TO_DIGIT_STRICT.get(c, c)
                elif i == 2:
                    if c.isdigit():
                        item['label'] = cls.DIGIT_TO_LETTER_STRICT.get(c, c)
                elif i == 3:
                    if c in ['O', 'Q', 'I', 'L']:
                        item['label'] = cls.LETTER_TO_DIGIT_STRICT.get(c, c)

            for item in bottom_items:
                c = item['label']
                item['label'] = cls.LETTER_TO_DIGIT_STRICT.get(c, c)

            return "".join([x['label'] for x in top_items]), "".join([x['label'] for x in bottom_items])
        else:
            total_len = len(top_items)
            for i, item in enumerate(top_items):
                c = item['label']
                if i in [0, 1]:
                    item['label'] = cls.LETTER_TO_DIGIT_STRICT.get(c, c)
                elif i == 2:
                    if c.isdigit():
                        item['label'] = cls.DIGIT_TO_LETTER_STRICT.get(c, c)
                elif i == 3 and total_len <= 8:
                    item['label'] = cls.LETTER_TO_DIGIT_STRICT.get(c, c)
                elif i >= 4:
                    item['label'] = cls.LETTER_TO_DIGIT_STRICT.get(c, c)

            return "".join([x['label'] for x in top_items]), ""

    @classmethod
    def format_from_lines(cls, top_line: list, bottom_line: list, is_two_lines: bool) -> str:
        if not is_two_lines:
            clean_top = "".join([x['label'] for x in top_line])
            if len(clean_top) >= 7:
                province = clean_top[:2]
                remainder = clean_top[2:]
                num_len = 5 if len(clean_top) >= 8 else 4
                series = remainder[:-num_len]
                num_seq = remainder[-num_len:]
                num_fmt = f"{num_seq[:3]}.{num_seq[3:]}" if len(num_seq) == 5 else num_seq
                return f"{province}{series}-{num_fmt}"
            return clean_top
        else:
            clean_top = "".join([x['label'] for x in top_line])
            clean_bot = "".join([x['label'] for x in bottom_line])
            bot_fmt = f"{clean_bot[:3]}.{clean_bot[3:]}" if len(clean_bot) == 5 else clean_bot
            if len(clean_top) == 4:
                return f"{clean_top[:2]}-{clean_top[2:]} {bot_fmt}"
            elif len(clean_top) == 3:
                return f"{clean_top} {bot_fmt}"
            return f"{clean_top} {bot_fmt}".strip()

    @classmethod
    def sort_and_format(cls, char_items, img_w, img_h):
        if not char_items:
            return "", [], [], [], False

        filtered = cls.filter_boxes_and_border_noise(char_items, img_w, img_h)
        if not filtered:
            return "", [], [], [], False

        is_two_lines = cls.is_really_two_lines(filtered)

        if not is_two_lines:
            filtered.sort(key=lambda x: x['cx'])
            if len(filtered) > 9:
                filtered = sorted(filtered, key=lambda x: x['conf'], reverse=True)[:9]
                filtered.sort(key=lambda x: x['cx'])

            filtered = cls._clean_one_line_leading_noise(filtered)
            cls._apply_vn_rules(filtered, [], is_two_lines=False)
            formatted = cls.format_from_lines(filtered, [], is_two_lines=False)
            return formatted, filtered, filtered, [], False
        else:
            top_line, bottom_line = cls.split_two_lines_sloped(filtered)
            if len(top_line) > 4:
                top_line = sorted(top_line, key=lambda x: x['conf'], reverse=True)[:4]
                top_line.sort(key=lambda x: x['cx'])
            if len(bottom_line) > 5:
                bottom_line = sorted(bottom_line, key=lambda x: x['conf'], reverse=True)[:5]
                bottom_line.sort(key=lambda x: x['cx'])

            cls._apply_vn_rules(top_line, bottom_line, is_two_lines=True)
            formatted = cls.format_from_lines(top_line, bottom_line, is_two_lines=True)
            return formatted, top_line + bottom_line, top_line, bottom_line, True


class OCREngine:
    """
    STAGE 2 - BỘ GIẢI MÃ KÝ TỰ SIÊU TỐC (EARLY-EXIT CASCADE + BATCH INFERENCE):
    - Tầng 1 (Fast Tier): Quét batch 2 biến thể chuẩn. Thoát sớm khi đủ 8-9 ký tự, đúng mã tỉnh, conf >= 83%.
    - Tầng 2 (Deep Tier): Quét batch NightBoost, AntiGlare, PerspectiveRectified, Deshear ở imgsz=512, conf=0.06.
    - Tầng 3 (Rotation Tier): Tự động thử xoay 180° / 90° khi chiều thẳng đọc kém.
    """

    def __init__(self):
        logger.info("=== NẠP LÕI YOLO OCR (EARLY-EXIT CASCADE + BATCH INFERENCE) ===")
        ocr_model_path = MODELS_DIR / OCR_MODEL_NAME
        self.model = YOLO(str(ocr_model_path)) if ocr_model_path.exists() else None
        self.class_names = self.model.names if self.model else []
        self.sorter = GeometrySorter()
        self.preprocessor = PlatePreprocessor()

    def _score_yolo_result(self, variant_name: str, img_variant: np.ndarray, yolo_res, imgsz: int) -> dict:
        img_h, img_w = img_variant.shape[:2]
        raw_items = []
        if yolo_res is not None and yolo_res.boxes is not None:
            for box in yolo_res.boxes:
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

        plate_text, valid_items, top_line, bottom_line, is_two_lines = self.sorter.sort_and_format(
            raw_items, img_w, img_h
        )
        avg_conf = (sum([x['conf'] for x in valid_items]) / len(valid_items) * 100.0) if valid_items else 0.0
        sum_conf = sum([x['conf'] for x in valid_items]) * 100.0

        char_count = len(valid_items)
        top_cnt = len(top_line)
        bot_cnt = len(bottom_line)
        score = avg_conf * 2.0 + sum_conf * 0.45

        if is_two_lines:
            if top_cnt == 4 and bot_cnt == 5:
                score += 600.0
            elif top_cnt == 3 and bot_cnt == 5:
                score += 540.0
            elif top_cnt in [3, 4] and bot_cnt == 4:
                score += 480.0
            elif top_cnt > bot_cnt:
                score -= 350.0
            else:
                score -= 200.0
        else:
            if char_count in [8, 9]:
                score += 540.0
            elif char_count == 7:
                score += 380.0
            else:
                score -= 250.0

        clean_alnum = "".join([c for c in plate_text if c.isalnum()])
        has_valid_prov = (len(clean_alnum) >= 2 and (clean_alnum[:2] in VALID_PROVINCE_CODES or clean_alnum[:2] == "80"))
        if len(clean_alnum) >= 2:
            score += 200.0 if has_valid_prov else -260.0

        if len(clean_alnum) >= 3 and clean_alnum[2].isalpha():
            score += 85.0

        if any(k in variant_name for k in ["NaturalSharp", "YellowToWhite_Priority", "CleanCLAHE", "NightBoost", "AntiGlare"]):
            score += 30.0

        return {
            'score': score,
            'text': plate_text,
            'items': valid_items,
            'top_line': top_line,
            'bottom_line': bottom_line,
            'is_two_lines': is_two_lines,
            'avg_conf': avg_conf,
            'img': img_variant,
            'char_count': char_count,
            'top_cnt': top_cnt,
            'has_valid_prov': has_valid_prov,
            'name': f"{variant_name}_{imgsz}"
        }

    def _run_batch_passes(self, named_variants: list, prefix: str, imgsz: int = 416, conf_thresh: float = 0.10) -> list:
        if not named_variants:
            return []
        imgs = [v_img for _, v_img in named_variants]
        results = self.model(imgs, imgsz=imgsz, verbose=False, conf=conf_thresh, iou=0.45)

        out = []
        for (v_name, v_img), res in zip(named_variants, results):
            out.append(self._score_yolo_result(f"{prefix}_{v_name}", v_img, res, imgsz))
        return out

    def _apply_selective_consensus(self, pass_results: list) -> dict:
        best_res = max(pass_results, key=lambda x: x['score'])
        if not best_res['items']:
            return best_res

        if best_res['avg_conf'] >= 82.0 and best_res['char_count'] >= 8:
            return best_res

        target_is_2l = best_res['is_two_lines']
        target_top_len = len(best_res['top_line'])
        target_bot_len = len(best_res['bottom_line'])
        target_prefix = best_res['name'].split('_')[0]

        strong_peers = [
            p for p in pass_results
            if p['name'].startswith(target_prefix)
            and p['is_two_lines'] == target_is_2l
            and len(p['top_line']) == target_top_len
            and len(p['bottom_line']) == target_bot_len
            and p['avg_conf'] >= 55.0
            and "Deshear" not in p['name']
        ]

        if len(strong_peers) < 3:
            return best_res

        voted_top = [dict(x) for x in best_res['top_line']]
        for idx in range(target_top_len):
            votes = defaultdict(float)
            for p in strong_peers:
                ch_item = p['top_line'][idx]
                votes[ch_item['label']] += (ch_item['conf'] ** 2)
            voted_top[idx]['label'] = max(votes.items(), key=lambda kv: kv[1])[0]

        voted_bottom = [dict(x) for x in best_res['bottom_line']]
        for idx in range(target_bot_len):
            votes = defaultdict(float)
            for p in strong_peers:
                ch_item = p['bottom_line'][idx]
                votes[ch_item['label']] += (ch_item['conf'] ** 2)
            voted_bottom[idx]['label'] = max(votes.items(), key=lambda kv: kv[1])[0]

        voted_text = self.sorter.format_from_lines(voted_top, voted_bottom, target_is_2l)
        clean_voted = "".join([c for c in voted_text if c.isalnum()])
        if len(clean_voted) >= 7 and (clean_voted[:2] in VALID_PROVINCE_CODES or clean_voted[:2] == "80"):
            best_res['text'] = voted_text
            best_res['top_line'] = voted_top
            best_res['bottom_line'] = voted_bottom
            best_res['items'] = voted_top + voted_bottom

        return best_res

    def _draw_debug_boxes(self, final_res: dict) -> np.ndarray:
        debug_img = final_res['img'].copy()
        for item in final_res['items']:
            x1, y1, x2, y2 = int(item['x1']), int(item['y1']), int(item['x2']), int(item['y2'])
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_img, item['label'], (x1 + 2, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA
            )
        return debug_img

    def extract_text(self, image: np.ndarray, orient: str = "ROT0") -> tuple:
        if image is None or image.size == 0 or self.model is None:
            return "", image, 0.0

        aligned, img_up, plate_type = self.preprocessor.prepare_base_image(image, orient=orient)

        # TẦNG 1: 
        fast_variants = self.preprocessor.generate_fast_variants(img_up, plate_type)
        pass_results = self._run_batch_passes(fast_variants, prefix="MAIN", imgsz=416, conf_thresh=0.10)
        best_fast = max(pass_results, key=lambda x: x['score'])

        
        is_complete_fast = (best_fast['char_count'] >= 9) or (not best_fast['is_two_lines'] and best_fast['char_count'] >= 8)
        if is_complete_fast and best_fast['has_valid_prov'] and best_fast['avg_conf'] >= 83.0:
            return best_fast['text'], self._draw_debug_boxes(best_fast), best_fast['avg_conf']

        # TẦNG 2: CHUYÊN SÂU (NightBoost, AntiGlare, Homography, Deshear ở imgsz=512, conf=0.06)
        deep_variants = self.preprocessor.generate_deep_variants(aligned, img_up, plate_type)
        if deep_variants:
            pass_results.extend(self._run_batch_passes(deep_variants, prefix="MAIN", imgsz=512, conf_thresh=0.06))

        best_main = max(pass_results, key=lambda x: x['score'])
        if best_main['char_count'] >= 7 and best_main['has_valid_prov'] and best_main['avg_conf'] >= 72.0:
            final_res = self._apply_selective_consensus(pass_results)
            return final_res['text'], self._draw_debug_boxes(final_res), final_res['avg_conf']

        # TẦNG 3: TỰ ĐỘNG XOAY 180° / 90° (Chỉ chạy khi chiều thẳng thực sự đọc kém)
        fallback_orients = []
        if orient == "ROT0":
            fallback_orients.append("ROT180")
            if best_main['char_count'] < 6 or not best_main['has_valid_prov']:
                fallback_orients.extend(["ROT90_CW", "ROT90_CCW"])
        else:
            fallback_orients.append("ROT0")

        for fb_orient in fallback_orients:
            _, fb_up, fb_type = self.preprocessor.prepare_base_image(image, orient=fb_orient)
            fb_fast = self.preprocessor.generate_fast_variants(fb_up, fb_type)
            pass_results.extend(self._run_batch_passes(fb_fast, prefix=f"FB_{fb_orient}", imgsz=416, conf_thresh=0.10))

        final_res = self._apply_selective_consensus(pass_results)
        return final_res['text'], self._draw_debug_boxes(final_res), final_res['avg_conf']
