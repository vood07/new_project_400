import cv2
import numpy as np
import pytesseract
from deskew import determine_skew
from PIL import Image
import os


class DocumentAligner:
    def __init__(self):
        self.debug = False

    def fix_orientation(self, image):
        """Етап 1: Грубий поворот 0/90/180/270 через Tesseract OSD"""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
            angle = osd.get('rotate', 0)

            if angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180), 180
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 270

            return image, 0
        except Exception as e:
            # Якщо OSD не спрацював (мало тексту), пробуємо альтернативу
            return self._fallback_orientation(image)

    def _fallback_orientation(self, image):
        """Альтернатива: шукаємо текстові блоки і їхню домінантну орієнтацію"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Шукаємо компоненти зв'язності
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Аналізуємо aspect ratio компонентів (текст зазвичай ширший за вищий)
        angles_to_test = [0, 90, 180, 270]
        best_score = -1
        best_angle = 0

        for angle in angles_to_test:
            test_img = self._rotate_simple(image, angle)
            score = self._estimate_text_score(test_img)
            if score > best_score:
                best_score = score
                best_angle = angle

        return self._rotate_simple(image, best_angle), best_angle

    def _estimate_text_score(self, image):
        """Оцінюємо "текстовість" зображення"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Горизонтальна проекція (текст має піки-рядки)
        projection = np.sum(gray < 128, axis=1)

        # Рахуємо варіацію (текст має чіткі рядки)
        variance = np.var(projection)

        # Перевіряємо aspect ratio (текст зазвичай landscape або portrait з текстом по ширині)
        aspect = w / h if h > 0 else 1

        # Комбінований скор
        return variance * (1 if 0.5 < aspect < 2 else 0.5)

    def _rotate_simple(self, image, angle):
        if angle == 0: return image
        if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def deskew(self, image):
        """Етап 2: Точне вирівнювання дрібного нахилу"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Спосіб 1: deskew бібліотека (найнадійніший для документів)
        try:
            angle = determine_skew(gray)
            if angle and abs(angle) > 0.1:
                return self._rotate_precise(image, -angle), angle
        except:
            pass

        # Спосіб 2: Hough Transform (як у твоєму коді)
        angle = self._hough_skew(gray)
        if abs(angle) > 0.1:
            return self._rotate_precise(image, -angle), angle

        return image, 0.0

    def _hough_skew(self, gray):
        """Знаходимо кут нахилу текстових рядків"""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 15)

        # Морфологія для з'єднання букв у рядки
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        lines_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(lines_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 15:  # Ігноруємо вертикальні лінії
                    angles.append(angle)

        return np.median(angles) if angles else 0.0

    def _rotate_precise(self, image, angle):
        """Пrecise rotation з розширенням canvas"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Обчислюємо новий розмір
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Коригуємо зсув
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    def perspective_correction(self, image):
        """Етап 3: Перспективна корекція (опціонально, якщо документ 'закошений')"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Знаходимо контур документа
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Беремо найбільший контур
        largest = max(contours, key=cv2.contourArea)

        # Апроксимуємо чотирикутником
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            return self._four_point_transform(image, approx.reshape(4, 2))

        # Якщо не знайшли 4 кути, пробуємо minAreaRect
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return self._four_point_transform(image, box)

    def _four_point_transform(self, image, pts):
        """Перспективна трансформація"""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (int(width), int(height)),
                                   borderValue=(255, 255, 255))

    def _order_points(self, pts):
        """Сортуємо точки: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def process(self, image_path, output_path=None):
        """Повний пайплайн"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не вдалося завантажити {image_path}")

        meta = {}

        # Етап 1: Орієнтація
        image, angle = self.fix_orientation(image)
        meta['orientation'] = angle

        # Етап 2: Deskew (тільки якщо орієнтація виправлена)
        image, skew = self.deskew(image)
        meta['deskew'] = skew

        # Етап 3: Перспектива (опціонально, для фото з камери)
        # image = self.perspective_correction(image)

        if output_path:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return image, meta


# --- ВИКОРИСТАННЯ ---
if __name__ == "__main__":
    import sys

    aligner = DocumentAligner()

    # Беремо шлях з аргументу або використовуємо дефолтний
    input_file = sys.argv[1] if len(sys.argv) > 1 else "128_.jpg"
    output_file = f"aligned_{os.path.basename(input_file)}"

    print(f"🔄 Обробка: {input_file}")
    print("-" * 40)

    try:
        # Завантажуємо для перевірки
        test_image = cv2.imread(input_file)
        if test_image is None:
            print(f"❌ Не вдалося завантажити {input_file}")
            print("Перевір, чи файл існує і чи це зображення")
            sys.exit(1)

        print(f"📐 Розмір: {test_image.shape[1]}x{test_image.shape[0]} пікселів")

        # Обробка
        aligned, meta = aligner.process(input_file, output_file)

        # Результати
        print(f"\n✅ ГОТОВО!")
        print(f"   Поворот: {meta['orientation']}°")
        print(f"   Нахил: {meta['deskew']:.2f}°")
        print(f"   Результат: {output_file}")

        # Показуємо (опціонально)
        show_comparison = True
        if show_comparison:
            # Створюємо порівняння
            orig = cv2.resize(test_image, (600, 800)) if test_image.shape[1] > 600 else test_image
            res = cv2.resize(aligned, (600, 800)) if aligned.shape[1] > 600 else aligned

            # Однакова висота для склейки
            h = min(orig.shape[0], res.shape[0])
            orig = orig[:h]
            res = res[:h]

            comparison = cv2.hconcat([orig, res])
            cv2.imshow("Оригінал | Вирівняне", comparison)
            print("\nНатисни будь-яку клавішу для закриття...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback

        traceback.print_exc()
