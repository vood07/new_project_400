import numpy as np
import cv2
import json
import os


class NumberLetterTest:
    def __init__(self, image_path, gradus=90):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Не вдалося завантажити зображення: {image_path}")

        self.original_height, self.original_width = self.image.shape[:2]
        self.gradus = gradus
        self.koord_zagal = []

        # Поворот зображення
        self.image, self.rotation_matrix = self._rotate_image(self.image, gradus)
        self.new_height, self.new_width = self.image.shape[:2]

    def _rotate_image(self, image, angle):
        """Поворот без обрізання"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        angle_rad = np.deg2rad(angle)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))

        new_width = int(height * sin_a + width * cos_a)
        new_height = int(height * cos_a + width * sin_a)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        tx = (new_width - width) // 2
        ty = (new_height - height) // 2
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return rotated, rotation_matrix

    def detect_cells(self, table_region=None, min_area=500, max_area=20000):
        """
        Виявлення клітинок таблиці через знаходження контурів.
        Повертає список {id, x, y, w, h}
        """
        # Область пошуку
        if table_region:
            x1, y1, x2, y2 = table_region
            roi = self.image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            roi = self.image.copy()
            offset_x, offset_y = 0, 0

        # Підготовка зображення
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Бінаризація - інвертуємо, щоб лінії були білими
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Морфологія для видалення шуму і з'єднання ліній
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Знаходимо всі контури
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # Фільтри за розміром і формою
            if min_area < area < max_area and w > 20 and h > 20:
                aspect_ratio = w / float(h)
                # Пропускаємо занадто витягнуті прямокутники (лінії)
                if 0.2 < aspect_ratio < 5.0:
                    cells.append({
                        'id': 0,  # тимчасове значення
                        'x': x + offset_x,
                        'y': y + offset_y,
                        'w': w,
                        'h': h
                    })

        # Сортування: зверху вниз, зліва направо
        cells.sort(key=lambda c: (c['y'], c['x']))

        # Перенумерація
        for i, cell in enumerate(cells, 1):
            cell['id'] = i

        self.koord_zagal = cells
        return cells

    def detect_by_lines(self, table_region=None, min_cell_size=30):
        """
        Альтернативний метод - через лінії таблиці.
        Краще працює для чітких таблиць.
        """
        if table_region:
            x1, y1, x2, y2 = table_region
            roi = self.image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            roi = self.image.copy()
            offset_x, offset_y = 0, 0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Виділяємо горизонтальні лінії
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Виділяємо вертикальні лінії
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Об'єднуємо
        table_structure = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)
        _, table_structure = cv2.threshold(table_structure, 50, 255, cv2.THRESH_BINARY)

        # Знаходимо перетини (клітинки)
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w > min_cell_size and h > min_cell_size and w < 300 and h < 300:
                # Перевіряємо, що це не лінія
                if w / h > 0.3 and h / w > 0.3:
                    cells.append({
                        'id': 0,
                        'x': x + offset_x,
                        'y': y + offset_y,
                        'w': w,
                        'h': h
                    })

        # Сортування і нумерація
        cells.sort(key=lambda c: (c['y'], c['x']))
        for i, cell in enumerate(cells, 1):
            cell['id'] = i

        self.koord_zagal = cells
        return cells

    def process(self, method='contours', table_region=None, **kwargs):
        """
        Головний метод обробки.
        method: 'contours' або 'lines'
        """
        if method == 'contours':
            return self.detect_cells(table_region=table_region, **kwargs)
        elif method == 'lines':
            return self.detect_by_lines(table_region=table_region, **kwargs)
        else:
            raise ValueError("method має бути 'contours' або 'lines'")

    def save_to_json(self, output_path="coordinates_output.json"):
        """Збереження у форматі data_for_team.json"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.koord_zagal, f, indent=4, ensure_ascii=False)
        print(f"Збережено {len(self.koord_zagal)} об'єктів у {output_path}")
        return output_path

    def save_rotated_image(self, output_path="rotated.jpg"):
        cv2.imwrite(output_path, self.image)
        print(f"Зображення збережено: {output_path}")
        return output_path

    def visualize(self, output_name="result.jpg", show_ids=True):
        vis = self.image.copy()
        for obj in self.koord_zagal:
            x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if show_ids:
                cv2.putText(vis, str(obj['id']), (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(output_name, vis)
        print(f"Візуалізацію збережено: {output_name}")
        return vis


# --- ГОЛОВНИЙ БЛОК ---

if __name__ == "__main__":
    # Шлях до фото (змініть на свій)
    PHOTO_PATH = "test1.jpeg"  # або повний шлях

    # Перевірка
    if not os.path.exists(PHOTO_PATH):
        print(f"ПОМИЛКА: Файл не знайдено: {PHOTO_PATH}")
        print(f"Поточна папка: {os.getcwd()}")
        print(f"Файли: {os.listdir('.')}")
        exit(1)

    # Створюємо об'єкт і повертаємо фото
    test = NumberLetterTest(PHOTO_PATH, gradus=-90)

    print(f"Оригінал: {test.original_width}x{test.original_height}")
    print(f"Після повороту: {test.new_width}x{test.new_height}")

    # Виявляємо клітинки
    print("\n--- Метод contours ---")
    coords = test.process(method='contours', min_area=800, max_area=15000)
    print(f"Знайдено: {len(coords)} клітинок")

    # Зберігаємо результати
    test.save_to_json("coordinates.json")
    test.save_rotated_image("rotated.jpg")
    test.visualize("result.jpg")

    # Пробуємо другий метод
    print("\n--- Метод lines ---")
    test2 = NumberLetterTest(PHOTO_PATH, gradus=90)
    coords2 = test2.process(method='lines', min_cell_size=40)
    print(f"Знайдено: {len(coords2)} клітинок")
    test2.save_to_json("coordinates_lines.json")
    test2.visualize("result_lines.jpg")

    print("\n--- ГОТОВО ---")