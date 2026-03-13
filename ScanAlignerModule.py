import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import importlib.util

# Імпортуємо бібліотеку deskew напряму (мінус конфлікт з локальним файлом)
spec = importlib.util.find_spec("deskew")
if spec is None:
    raise ImportError("Бібліотека deskew не встановлена. Виконай: pip install deskew")

deskew_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deskew_module)
determine_skew = deskew_module.determine_skew


class ScanAlignerModule:
    """
    Модуль вирівнювання сканів для Orchestrator/Workflow Engine.
    Отримує скан → вирівнює → передає наступним 35 модулям.
    """

    def __init__(self):
        self.name = "ScanAligner"
        self.next_modules = []  # Сюди додаси свої 35 модулів

    def register_next_module(self, module):
        """Реєстрація наступного модуля в ланцюгу"""
        self.next_modules.append(module)

    def process(self, image_data, metadata):
        """
        Головний метод для Orchestrator.

        Args:
            image_data: numpy array (скан від File Ingestion)
            metadata: dict з meta-інформацією (ID особи, шаблон, тощо)

        Returns:
            tuple: (aligned_image, updated_metadata, success_flag)
        """
        try:
            # === ЕТАП 1: Вирівнювання орієнтації ===
            aligned, orientation_meta = self._fix_orientation(image_data)

            # === ЕТАП 2: Вирівнювання нахилу ===
            aligned, deskew_meta = self._deskew(aligned)

            # Оновлюємо метадані
            metadata.update({
                'rotation_applied': orientation_meta['angle'],
                'deskew_applied': deskew_meta['angle'],
                'aligner_status': 'success'
            })

            # === ЕТАП 3: Передача 35 модулям ===
            self._dispatch_to_modules(aligned, metadata)

            return aligned, metadata, True

        except Exception as e:
            metadata['aligner_error'] = str(e)
            metadata['aligner_status'] = 'failed'
            return image_data, metadata, False

    def _fix_orientation(self, image):
        """Визначає і виправляє грубий поворот 0/90/180/270"""
        try:
            # Tesseract OSD
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
            angle = osd.get('rotate', 0)

            # Поворот
            if angle == 90:
                result = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                result = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                result = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                result = image

            return result, {'angle': angle, 'method': 'tesseract_osd'}

        except Exception:
            # Fallback: аналіз проекцій
            return self._fallback_orientation(image)

    def _fallback_orientation(self, image):
        """Альтернатива без Tesseract"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_score = -1
        best_angle = 0

        for angle in [0, 90, 180, 270]:
            test_img = self._rotate_simple(image, angle)
            score = self._estimate_text_score(test_img)
            if score > best_score:
                best_score = score
                best_angle = angle

        result = self._rotate_simple(image, best_angle)
        return result, {'angle': best_angle, 'method': 'projection_analysis'}

    def _estimate_text_score(self, image):
        """Оцінка "текстовості" — чим більше рядків, тим краще"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        projection = np.sum(gray < 128, axis=1)
        return np.var(projection)

    def _rotate_simple(self, image, angle):
        """Поворот на 90/180/270"""
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _deskew(self, image):
        """Вирівнювання дрібного нахилу"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Спосіб 1: бібліотека deskew
        try:
            angle = determine_skew(gray)
            if angle and abs(angle) > 0.1:
                result = self._rotate_precise(image, -angle)
                return result, {'angle': angle, 'method': 'deskew_lib'}
        except:
            pass

        # Спосіб 2: Hough Transform
        angle = self._hough_skew(gray)
        if abs(angle) > 0.1:
            result = self._rotate_precise(image, -angle)
            return result, {'angle': angle, 'method': 'hough_transform'}

        return image, {'angle': 0, 'method': 'none'}

    def _hough_skew(self, gray):
        """Знаходження кута через лінії тексту"""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 15)

        # Об'єднуємо букви в рядки
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        lines_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Шукаємо лінії
        edges = cv2.Canny(lines_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 15:
                    angles.append(angle)

        return np.median(angles) if angles else 0.0

    def _rotate_precise(self, image, angle):
        """Точний поворот з розширенням полотна"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Новий розмір
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Центруємо
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    def _dispatch_to_modules(self, image, metadata):
        """
        Передача вирівняного зображення 35 модулям.
        Кожен модуль отримує (image, metadata) і обробляє синхронно або асинхронно.
        """
        for module in self.next_modules:
            try:
                # Синхронна передача
                module.receive(image.copy(), metadata.copy())
            except Exception as e:
                # Логування помилки конкретного модуля
                print(f"   ⚠️ Модуль {module.name} помилка: {e}")


# === ПРИКЛАД ІНТЕГРАЦІЇ З ТВОЄЮ СИСТЕМОЮ ===

class ExampleNextModule:
    """Приклад одного з 35 модулів"""

    def __init__(self, name):
        self.name = name

    def receive(self, image, metadata):
        # Тут твоя логіка: OCR, аналіз, тощо
        print(f"   📥 {self.name} отримав зображення {image.shape}")


class Orchestrator:
    """Твій блок керування (спрощено)"""

    def __init__(self):
        self.aligner = ScanAlignerModule()
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Реєстрація 35 модулів"""
        for i in range(1, 36):
            module = ExampleNextModule(f"Module_{i}")
            self.aligner.register_next_module(module)

    def process_file(self, file_path, person_id, template_id):
        """
        Головний метод виклику з File Ingestion
        """
        # Завантаження
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Не вдалося завантажити {file_path}")

        # Метадані з твоєї системи
        metadata = {
            'file_path': file_path,
            'person_id': person_id,
            'template_id': template_id,
            'timestamp': cv2.getTickCount()
        }

        # Вирівнювання + автоматична передача 35 модулям
        aligned, updated_meta, success = self.aligner.process(image, metadata)

        return {
            'success': success,
            'metadata': updated_meta,
            'aligned_image': aligned if not success else None
        }


# === ЗАПУСК ===
if __name__ == "__main__":
    # Ініціалізація системи
    orchestrator = Orchestrator()

    # Обробка файлу (викликається з File Ingestion)
    result = orchestrator.process_file(
        file_path="scan_001.jpg",
        person_id="P-12345",
        template_id="TEMPLATE_A"
    )

    print(f"\nРезультат: {result['metadata']}")