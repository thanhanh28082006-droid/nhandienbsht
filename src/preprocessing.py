import cv2
import numpy as np


class PlatePreprocessor:

    @staticmethod
    def detect_plate_color_type(image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return 'WHITE'

        h, w = image.shape[:2]
        cy1, cy2 = int(h * 0.18), int(h * 0.82)
        cx1, cx2 = int(w * 0.12), int(w * 0.88)
        core = image[cy1:cy2, cx1:cx2]
        if core.size == 0:
            core = image

        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)

        mask_yellow = cv2.inRange(hsv, np.array([8, 38, 50]), np.array([42, 255, 255]))
        yellow_ratio = np.count_nonzero(mask_yellow) / float(core.shape[0] * core.shape[1])

        mask_blue = cv2.inRange(hsv, np.array([95, 65, 40]), np.array([138, 255, 255]))
        blue_ratio = np.count_nonzero(mask_blue) / float(core.shape[0] * core.shape[1])

        mask_red1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 70, 40]), np.array([180, 255, 255]))
        red_ratio = (np.count_nonzero(mask_red1) + np.count_nonzero(mask_red2)) / float(core.shape[0] * core.shape[1])

        if blue_ratio > 0.22 or red_ratio > 0.22:
            return 'BLUE_RED'
        if yellow_ratio > 0.16:
            return 'YELLOW'

        b, g, r = cv2.split(core)
        if (np.mean(r) - np.mean(b) > 22) and (np.mean(g) - np.mean(b) > 16):
            return 'YELLOW'

        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        if np.median(gray) < 90 and np.percentile(gray, 90) > 165:
            return 'BLUE_RED'

        return 'WHITE'

    @staticmethod
    def gentle_deskew(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 45, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180.0,
                threshold=24,
                minLineLength=int(w * 0.32),
                maxLineGap=10
            )
            if lines is not None:
                angles = []
                for line in lines:
                    if isinstance(line, np.ndarray) and len(line) > 0:
                        coords = line[0] if len(line.shape) > 1 else line
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                            if x2 - x1 != 0:
                                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                                if -22.0 <= angle <= 22.0:
                                    angles.append(angle)
                if angles:
                    med_angle = float(np.median(angles))
                    if abs(med_angle) > 1.2:
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), med_angle, 1.0)
                        image = cv2.warpAffine(
                            image, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE
                        )
        except Exception:
            pass
        return image

    @staticmethod
    def apply_deshear_and_rotate(image: np.ndarray, shear_x: float = 0.0, angle: float = 0.0) -> np.ndarray:

        h, w = image.shape[:2]
        out = image.copy()
        if abs(angle) > 0.5:
            M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            out = cv2.warpAffine(
                out, M_rot, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(135, 135, 135)
            )
        if abs(shear_x) > 0.01:
            M_shear = np.float32([
                [1.0, shear_x, -shear_x * h / 2.0],
                [0.0, 1.0, 0.0]
            ])
            out = cv2.warpAffine(
                out, M_shear, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(135, 135, 135)
            )
        return out

    @staticmethod
    def extract_red_green_clean_plate(image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image)
        rg_gray = cv2.addWeighted(r, 0.75, g, 0.25, 0)

        p2, p98 = np.percentile(rg_gray, (2, 98))
        if p98 - p2 > 15:
            rg_stretched = np.clip((rg_gray.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        else:
            rg_stretched = cv2.normalize(rg_gray, None, 0, 255, cv2.NORM_MINMAX)

        smooth = cv2.bilateralFilter(rg_stretched, 5, 40, 40)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.subtract(smooth, cv2.multiply(blackhat, 0.45))

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
        clean_gray = clahe.apply(enhanced)

        blur = cv2.GaussianBlur(clean_gray, (0, 0), 1.1)
        sharp = cv2.addWeighted(clean_gray, 1.5, blur, -0.5, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    @classmethod
    def generate_variants(cls, image: np.ndarray) -> list:
        if image is None or image.size == 0:
            return []

        plate_type = cls.detect_plate_color_type(image)
        img_deskew = cls.gentle_deskew(image)
        h, w = img_deskew.shape[:2]

        ratio = w / float(h) if h > 0 else 1.0
        if ratio < 1.28:
            img_deskew = cv2.resize(img_deskew, (int(h * 1.45), h), interpolation=cv2.INTER_CUBIC)
            h, w = img_deskew.shape[:2]

        pad_y = max(12, int(h * 0.12))
        pad_x = max(16, int(w * 0.12))
        img_pad = cv2.copyMakeBorder(
            img_deskew, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=(138, 138, 138)
        )

        hp, wp = img_pad.shape[:2]
        scale = 4.6 if hp < 55 else (3.6 if hp < 85 else (2.6 if hp < 130 else 1.8))
        img_up = cv2.resize(img_pad, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        variants = []

        if plate_type == 'BLUE_RED':
            b, g, r = cv2.split(img_up)
            max_ch = np.maximum(np.maximum(r, g), b)
            inv_gray = 255 - max_ch
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
            v_inv1 = cv2.cvtColor(clahe.apply(inv_gray), cv2.COLOR_GRAY2BGR)
            variants.append(("V1_BlueRed_Inverted", v_inv1))

            blur = cv2.GaussianBlur(v_inv1, (0, 0), 1.3)
            v_inv2 = cv2.addWeighted(v_inv1, 1.5, blur, -0.5, 0)
            variants.append(("V2_BlueRed_Sharp", v_inv2))
            variants.append(("V3_Natural", img_up))
            return variants

        blur_nat = cv2.GaussianBlur(img_up, (0, 0), 1.2)
        v1_natural = cv2.addWeighted(img_up, 1.45, blur_nat, -0.45, 0)
        variants.append(("V1_NaturalSharp", v1_natural))

        v2_rg_clean = cls.extract_red_green_clean_plate(img_up)
        tag_rg = "V2_YellowToWhite_Priority" if plate_type == 'YELLOW' else "V2_RedGreenClean"
        variants.append((tag_rg, v2_rg_clean))

        lab = cv2.cvtColor(img_up, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(6, 6))
        l_eq = clahe.apply(l)
        v3_clahe = cv2.cvtColor(cv2.merge((l_eq, a, b_ch)), cv2.COLOR_LAB2BGR)
        blur_c = cv2.GaussianBlur(v3_clahe, (0, 0), 1.2)
        v3_clahe = cv2.addWeighted(v3_clahe, 1.45, blur_c, -0.45, 0)
        variants.append(("V3_NaturalCLAHE", v3_clahe))

        gray_clean = cv2.cvtColor(v3_clahe, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray_clean, 7, 50, 50)
        norm_gray = cv2.normalize(bilateral, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        blur_hb = cv2.GaussianBlur(norm_gray, (0, 0), 1.7)
        high_boost = cv2.addWeighted(norm_gray, 1.85, blur_hb, -0.85, 0)
        v4_deblur = cv2.cvtColor(high_boost, cv2.COLOR_GRAY2BGR)
        variants.append(("V4_FarDeblur", v4_deblur))

        base_for_shear = v2_rg_clean if plate_type == 'YELLOW' else v3_clahe
        v5_deshear_pos = cls.apply_deshear_and_rotate(base_for_shear, shear_x=0.17, angle=-2.5)
        variants.append(("V5_DeshearPos", v5_deshear_pos))

        v6_deshear_neg = cls.apply_deshear_and_rotate(base_for_shear, shear_x=-0.17, angle=2.5)
        variants.append(("V6_DeshearNeg", v6_deshear_neg))

        return variants