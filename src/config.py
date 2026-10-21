import logging
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TEST_IMAGES_DIR = BASE_DIR / "test_images"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_NAME = "yolo_detector.pt"
OCR_MODEL_NAME = "ocr_yolo.pt"


PROVINCE_MAP = {
    "11": "Cao Bang",
    "12": "Lang Son",
    "14": "Quang Ninh",
    "15": "Hai Phong", "16": "Hai Phong",
    "17": "Thai Binh",
    "18": "Nam Dinh",
    "19": "Phu Tho",
    "20": "Thai Nguyen",
    "21": "Yen Bai",
    "22": "Tuyen Quang",
    "23": "Ha Giang",
    "24": "Lao Cai",
    "25": "Lai Chau",
    "26": "Son La",
    "27": "Dien Bien",
    "28": "Hoa Binh",
    "29": "Ha Noi", "30": "Ha Noi", "31": "Ha Noi", "32": "Ha Noi", "33": "Ha Noi", "40": "Ha Noi",
    "34": "Hai Duong",
    "35": "Ninh Binh",
    "36": "Thanh Hoa",
    "37": "Nghe An",
    "38": "Ha Tinh",
    "39": "Dong Nai", "60": "Dong Nai",
    "41": "TP. Ho Chi Minh", "50": "TP. Ho Chi Minh", "51": "TP. Ho Chi Minh",
    "52": "TP. Ho Chi Minh", "53": "TP. Ho Chi Minh", "54": "TP. Ho Chi Minh",
    "55": "TP. Ho Chi Minh", "56": "TP. Ho Chi Minh", "57": "TP. Ho Chi Minh",
    "58": "TP. Ho Chi Minh", "59": "TP. Ho Chi Minh",
    "43": "Da Nang",
    "47": "Dak Lak",
    "48": "Dak Nong",
    "49": "Lam Dong",
    "61": "Binh Duong",
    "62": "Long An",
    "63": "Tien Giang",
    "64": "Vinh Long",
    "65": "Can Tho",
    "66": "Dong Thap",
    "67": "An Giang",
    "68": "Kien Giang",
    "69": "Ca Mau",
    "70": "Tay Ninh",
    "71": "Ben Tre",
    "72": "Ba Ria - Vung Tau",
    "73": "Quang Binh",
    "74": "Quang Tri",
    "75": "Thua Thien Hue",
    "76": "Quang Ngai",
    "77": "Binh Dinh",
    "78": "Phu Yen",
    "79": "Khanh Hoa",
    "80": "Cuc CSGT (Trung Uong)",
    "81": "Gia Lai",
    "82": "Kon Tum",
    "83": "Soc Trang",
    "84": "Tra Vinh",
    "85": "Ninh Thuan",
    "86": "Binh Thuan",
    "88": "Vinh Phuc",
    "89": "Hung Yen",
    "90": "Ha Nam",
    "92": "Quang Nam",
    "93": "Binh Phuoc",
    "94": "Bac Lieu",
    "95": "Hau Giang",
    "97": "Bac Kan",
    "98": "Bac Giang",
    "99": "Bac Ninh"
}

VALID_PROVINCE_CODES = set(PROVINCE_MAP.keys())


def get_province_name(plate_text: str) -> str:
    if not plate_text:
        return "Khong xac dinh"

    clean_text = "".join([c for c in plate_text if c.isalnum()])
    if len(clean_text) < 2:
        return "Khong xac dinh"

    prefix = clean_text[:2]
    return PROVINCE_MAP.get(prefix, "Khong xac dinh")



def setup_logger(name: str = "ALPR") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger