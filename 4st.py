import numpy as np
import cv2


class NumberLetterTest:
    def __init__(self, dictionary_foto, gradus=0):
        self.image = self._extract_image(dictionary_foto)
        self.gradus = gradus
        self.koord_zagal = []

    def _extract_image(self, dictionary_foto):
        for key, value in dictionary_foto.items():
            if key.lower().endswith('.jpg'):
                img = cv2.imread(value)
                return img
        raise ValueError("Файл не знайдено")

    def _get_projected_bounds(self, projection, min_dist=2, threshold=0):
        bounds = []
        start = None
        for i, val in enumerate(projection):
            if val > threshold and start is None:
                start = i
            elif val <= threshold and start is not None:
                if i - start >= min_dist:
                    bounds.append((start, i))
                start = None
        return bounds

    def process_photo(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 10)

        # Роз'єднуємо символи, що торкаються
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)

        h_proj = np.sum(binary, axis=1)
        row_bounds = self._get_projected_bounds(h_proj, min_dist=5, threshold=gray.shape[1] * 0.005)

        self.koord_zagal = []
        global_id = 1

        # Якщо ширина блоку більше 60 пікселів — це явно кілька символів
        MAX_CELL_WIDTH = 60

        for r_idx, (y_start, y_end) in enumerate(row_bounds):
            row_roi = binary[y_start:y_end, :]
            v_proj = np.sum(row_roi, axis=0)
            col_bounds = self._get_projected_bounds(v_proj, min_dist=2, threshold=0)

            for (x_start, x_end) in col_bounds:
                w = x_end - x_start
                h = y_end - y_start

                # Логіка автоматичного розрізання
                if w > MAX_CELL_WIDTH:
                    num_parts = int(round(w / (MAX_CELL_WIDTH * 0.8)))
                    if num_parts > 1:
                        part_w = w / num_parts
                        for n in range(num_parts):
                            new_x = x_start + (n * part_w)
                            self.koord_zagal.append({
                                'id': global_id,
                                'bbox': (int(new_x), int(y_start), int(part_w), int(h)),
                                'gradus': self.gradus
                            })
                            global_id += 1
                        continue

                self.koord_zagal.append({
                    'id': global_id,
                    'bbox': (int(x_start), int(y_start), int(w), int(h)),
                    'gradus': self.gradus
                })
                global_id += 1

        return self.koord_zagal

    def visualize(self, output_name="result_visual.jpg"):
        vis = self.image.copy()
        for obj in self.koord_zagal:
            x, y, w, h = obj['bbox']
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite(output_name, vis)


# --- ЗАПУСК І ОТРИМАННЯ МАСИВУ ---

dict_f = {"test.jpg": "PHOTO-2026-01-16-12-36-29.jpg"}
test = NumberLetterTest(dict_f, gradus=0)

# 1. Запускаємо обробку і зберігаємо результат у змінну
koord_zagal = test.process_photo()

# 2. Виводимо весь масив у консоль (Python List of Dictionaries)
print("--- ФІНАЛЬНИЙ МАСИВ КООРДИНАТ ---")
for obj in koord_zagal:
    print(obj)

# 3. Додатково виводимо кількість знайдених об'єктів
print(f"\nВсього об'єктів у масиві: {len(koord_zagal)}")

# 4. Зберігаємо картинку для візуального контролю
test.visualize()