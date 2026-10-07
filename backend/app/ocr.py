from pathlib import Path

import cv2
import numpy as np

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
    return _ocr


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def extract_image_text(image_bytes: bytes) -> str:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded image could not be decoded")
    result = get_ocr().ocr(preprocess(image), cls=True)
    items: list[tuple[float, float, str]] = []
    for page in result or []:
        for item in page or []:
            if item and len(item) > 1 and item[1]:
                box, text_info = item
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                items.append((sum(xs) / len(xs), sum(ys) / len(ys), text_info[0]))
    if not items:
        return ""

    image_width = image.shape[1]
    left = [item for item in items if item[0] < image_width * 0.47]
    right = [item for item in items if item[0] >= image_width * 0.53]
    has_two_columns = len(left) >= 4 and len(right) >= 4
    ordered = (left + right) if has_two_columns else items
    ordered.sort(key=lambda item: (0 if has_two_columns and item in left else 1, item[1], item[0]))
    return "\n".join(item[2] for item in ordered).strip()


def extract_pdf_text(file_path: Path, max_pages: int) -> str:
    import pymupdf

    pages: list[str] = []
    with pymupdf.open(file_path) as pdf:
        if len(pdf) > max_pages:
            raise ValueError(f"PDF exceeds the {max_pages}-page limit")
        for page in pdf:
            blocks = [
                block for block in page.get_text("blocks")
                if len(block) >= 5 and block[4].strip()
            ]
            if blocks:
                page_width = page.rect.width
                left = [block for block in blocks if (block[0] + block[2]) / 2 < page_width * 0.5]
                right = [block for block in blocks if (block[0] + block[2]) / 2 >= page_width * 0.5]
                two_columns = len(left) >= 2 and len(right) >= 2
                columns = [left, right] if two_columns else [blocks]
                ordered = [
                    block for column in columns
                    for block in sorted(column, key=lambda item: (item[1], item[0]))
                ]
                pages.append("\n".join(block[4].strip() for block in ordered))
            else:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                pages.append(extract_image_text(pixmap.tobytes("png")))
    return "\n\n".join(page for page in pages if page).strip()
