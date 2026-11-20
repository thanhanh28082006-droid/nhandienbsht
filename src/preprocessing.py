import cv2
import numpy as np


class PlatePreprocessor:
    """
    BỘ TIỀN XỬ LÝ ĐA PHỔ TỐI ƯU TỐC ĐỘ & CẮT SÁT SAU XOAY (LAZY MULTI-TIER PREPROCESSOR):
    - Tự động xoay phẳng góc nghiêng lớn (±42°) và cắt gọn phần viền thừa sau khi xoay
      để biển số luôn chiếm trọn khung hình (khắc phục hoàn toàn lỗi 30E-860.44).
    - Tầng Nhanh (`generate_fast_variants`): 2 biến thể tinh gọn cho ảnh rõ nét.
    - Tầng Chuyên Sâu (`generate_deep_variants`): NightBoost, AntiGlare, Homography, Deshear.
    """

    @staticmethod
    def align_box_orientation(image: np.ndarray, orient: str = "ROT0") -> np.ndarray:
        if image is None or image.size == 0:
            return image
        if orient == "ROT90_CW":
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orient == "ROT90_CCW":
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orient == "ROT180":
            return cv2.rotate(image, cv2.ROTATE_180)
        return image

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
        area = float(core.shape[0] * core.shape[1])

        mask_yellow = cv2.inRange(hsv, np.array([8, 36, 48]), np.array([43, 255, 255]))
        if np.count_nonzero(mask_yellow) / area > 0.16:
            return 'YELLOW'

        mask_blue = cv2.inRange(hsv, np.array([94, 60, 40]), np.array([138, 255, 255]))
        if np.count_nonzero(mask_blue) / area > 0.22:
            return 'BLUE_RED'

        mask_red1 = cv2.inRange(hsv, np.array([0, 65, 40]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 65, 40]), np.array([180, 255, 255]))
        if (np.count_nonzero(mask_red1) + np.count_nonzero(mask_red2)) / area > 0.22:
            return 'BLUE_RED'

        b, g, r = cv2.split(core)
        if (np.mean(r) - np.mean(b) > 22) and (np.mean(g) - np.mean(b) > 16):
            return 'YELLOW'

        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        if np.median(gray) < 88 and np.percentile(gray, 90) > 165:
            return 'BLUE_RED'

        return 'WHITE'

    @staticmethod
    def _crop_rotated_content(rotated_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Cắt bỏ các mảng tam giác viền xám thừa sau khi xoay ảnh góc lớn (như 30E-860.44),
        giữ lại phần lõi biển số có độ tương phản/sáng rõ nhất để không bị thu nhỏ chữ.
        """
        try:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return rotated_img
            x1, x2 = int(np.min(xs)), int(np.max(xs))
            y1, y2 = int(np.min(ys)), int(np.max(ys))
            cropped = rotated_img[y1:y2 + 1, x1:x2 + 1]
            if cropped.shape[0] < 16 or cropped.shape[1] < 24:
                return rotated_img

            # Nếu vùng xoay bị dính nền tối lớn phía trên/dưới (cản xe đen), thu gọn theo vùng sáng/cạnh
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            row_energy = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=1)
            thresh_e = np.max(row_energy) * 0.22
            active_rows = np.where(row_energy >= thresh_e)[0]
            if len(active_rows) > 12:
                ry1 = max(0, int(active_rows[0]) - 6)
                ry2 = min(cropped.shape[0], int(active_rows[-1]) + 7)
                if (ry2 - ry1) >= cropped.shape[0] * 0.30:
                    cropped = cropped[ry1:ry2, :]
            return cropped
        except Exception:
            return rotated_img

    @classmethod
    def rotate_bound_and_tight_crop(cls, image: np.ndarray, angle_deg: float) -> np.ndarray:
        if image is None or image.size == 0 or abs(angle_deg) < 0.8:
            return image
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos_v = abs(M[0, 0])
        sin_v = abs(M[0, 1])
        new_w = int((h * sin_v) + (w * cos_v))
        new_h = int((h * cos_v) + (w * sin_v))
        M[0, 2] += (new_w / 2.0) - cx
        M[1, 2] += (new_h / 2.0) - cy

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(138, 138, 138)
        )
        mask = np.ones((h, w), dtype=np.uint8) * 255
        rot_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
        return cls._crop_rotated_content(rotated, rot_mask)

    @classmethod
    def wide_angle_deskew(cls, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 45, 145, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180.0,
                threshold=18,
                minLineLength=int(min(w, h) * 0.30),
                maxLineGap=12
            )
            angles = []
            if lines is not None:
                for line in lines:
                    coords = line[0] if len(line.shape) > 1 else line
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        if x2 - x1 != 0:
                            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                            if -42.0 <= ang <= 42.0:
                                angles.append(ang)
            if angles:
                med_angle = float(np.median(angles))
                if abs(med_angle) > 1.8:
                    return cls.rotate_bound_and_tight_crop(image, med_angle)
        except Exception:
            pass
        return image

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @classmethod
    def perspective_rectify(cls, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        try:
            h, w = image.shape[:2]
            if h < 24 or w < 24:
                return image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 45, 145)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return image

            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            img_area = float(h * w)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area * 0.30 or area > img_area * 0.96:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype("float32")
                    rect = cls._order_points(pts)
                    (tl, tr, br, bl) = rect
                    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                    if max_w > w * 0.50 and max_h > h * 0.45:
                        dst = np.array([
                            [0, 0], [max_w - 1, 0],
                            [max_w - 1, max_h - 1], [0, max_h - 1]
                        ], dtype="float32")
                        M = cv2.getPerspectiveTransform(rect, dst)
                        return cv2.warpPerspective(
                            image, M, (max_w, max_h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE
                        )
        except Exception:
            pass
        return image

    @staticmethod
    def apply_affine_transform(image: np.ndarray, shear_x: float = 0.0, angle: float = 0.0) -> np.ndarray:
        h, w = image.shape[:2]
        out = image.copy()
        if abs(angle) > 0.3:
            M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            out = cv2.warpAffine(
                out, M_rot, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(138, 138, 138)
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
                borderValue=(138, 138, 138)
            )
        return out

    @staticmethod
    def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / max(0.08, gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.float32)
        lut = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(image, lut)

    @staticmethod
    def denoise_and_enhance_details(image: np.ndarray, clip_limit: float = 2.6, sharp_gain: float = 1.45) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        denoise = cv2.bilateralFilter(gray, 5, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(6, 6))
        eq = clahe.apply(denoise)
        gauss = cv2.GaussianBlur(eq, (0, 0), 1.2)
        sharp = cv2.addWeighted(eq, sharp_gain, gauss, -(sharp_gain - 1.0), 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

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
        enhanced = cv2.subtract(smooth, cv2.multiply(blackhat, 0.35))

        clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(6, 6))
        clean_gray = clahe.apply(enhanced)
        blur = cv2.GaussianBlur(clean_gray, (0, 0), 1.1)
        sharp = cv2.addWeighted(clean_gray, 1.45, blur, -0.45, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    @classmethod
    def prepare_base_image(cls, image: np.ndarray, orient: str = "ROT0") -> tuple:
        aligned = cls.align_box_orientation(image, orient)
        plate_type = cls.detect_plate_color_type(aligned)
        img_deskew = cls.wide_angle_deskew(aligned)
        h, w = img_deskew.shape[:2]

        pad_y = max(12, int(h * 0.12))
        pad_x = max(16, int(w * 0.12))
        img_pad = cv2.copyMakeBorder(
            img_deskew, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=(138, 138, 138)
        )

        hp = img_pad.shape[0]
        scale = 3.6 if hp < 50 else (2.8 if hp < 85 else (2.0 if hp < 130 else 1.5))
        img_up = cv2.resize(img_pad, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return aligned, img_up, plate_type

    @classmethod
    def generate_fast_variants(cls, img_up: np.ndarray, plate_type: str) -> list:
        variants = []
        if plate_type == 'BLUE_RED':
            b, g, r = cv2.split(img_up)
            inv_gray = 255 - np.maximum(np.maximum(r, g), b)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
            v_inv = cv2.cvtColor(clahe.apply(inv_gray), cv2.COLOR_GRAY2BGR)
            variants.append(("V1_BlueRed_Inverted", v_inv))
            variants.append(("V2_Natural", img_up))
            return variants

        blur_nat = cv2.GaussianBlur(img_up, (0, 0), 1.1)
        v1_natural = cv2.addWeighted(img_up, 1.38, blur_nat, -0.38, 0)
        variants.append(("V1_NaturalSharp", v1_natural))

        if plate_type == 'YELLOW':
            variants.append(("V2_YellowToWhite_Priority", cls.extract_red_green_clean_plate(img_up)))
        else:
            variants.append(("V2_CleanCLAHE", cls.denoise_and_enhance_details(img_up, clip_limit=2.6, sharp_gain=1.45)))

        return variants

    @classmethod
    def generate_deep_variants(cls, aligned: np.ndarray, img_up: np.ndarray, plate_type: str) -> list:
        variants = []
        if plate_type == 'BLUE_RED':
            b, g, r = cv2.split(img_up)
            inv_gray = 255 - np.maximum(np.maximum(r, g), b)
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(6, 6))
            v_inv = cv2.cvtColor(clahe.apply(inv_gray), cv2.COLOR_GRAY2BGR)
            blur = cv2.GaussianBlur(v_inv, (0, 0), 1.2)
            variants.append(("V3_BlueRed_Sharp", cv2.addWeighted(v_inv, 1.5, blur, -0.5, 0)))
            return variants

        v_clahe = cls.denoise_and_enhance_details(img_up, clip_limit=2.8, sharp_gain=1.50)

        if plate_type != 'YELLOW':
            variants.append(("V3_RedGreenClean", cls.extract_red_green_clean_plate(img_up)))

        variants.append(("V4_NightBoost", cls.apply_gamma(v_clahe, gamma=1.90)))

        variants.append(("V5_AntiGlare", cls.apply_gamma(v_clahe, gamma=0.58)))

        rectified = cls.perspective_rectify(aligned)
        if rectified is not None and rectified.size > 0:
            rh, rw = rectified.shape[:2]
            r_pad = cv2.copyMakeBorder(
                rectified, int(rh * 0.12), int(rh * 0.12), int(rw * 0.12), int(rw * 0.12),
                cv2.BORDER_CONSTANT, value=(138, 138, 138)
            )
            r_up = cv2.resize(r_pad, (img_up.shape[1], img_up.shape[0]), interpolation=cv2.INTER_CUBIC)
            variants.append(("V6_PerspectiveRectified", r_up))

        variants.append(("V7_DeshearPos", cls.apply_affine_transform(v_clahe, shear_x=0.14, angle=-2.0)))
        variants.append(("V8_DeshearNeg", cls.apply_affine_transform(v_clahe, shear_x=-0.14, angle=2.0)))

        return variants
