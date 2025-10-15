import os
import sane
import cv2  # OpenCV do analizy obrazu
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance

# --- Konfiguracja ---
NAZWA_SKANERA = "fujitsu:fi-630dj:13583" # Wpisz nazwę swojego skanera
DPI = 300

# Konfiguracja Poprawy Obrazu
WSP_JASNOSCI = 1.05  # 1.0 = bez zmian, > 1.0 = jaśniej
WSP_KONTRASTU = 1.15  # 1.0 = bez zmian, > 1.0 = większy kontrast
WSP_NASYCENIA = 1.10  # 1.0 = bez zmian, > 1.0 = żywsze kolory

# --- Koniec Konfiguracji ---

def order_points(pts):
    """Porządkuje 4 rogi prostokąta: lewy-górny, prawy-górny, prawy-dolny, lewy-dolny."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_image(image_path):
    """
    Odnajduje kartę na obrazie, wycina ją, prostuje i poprawia kolory.
    Zwraca przetworzony obraz jako obiekt PIL.Image lub None, jeśli nie znaleziono karty.
    """
    print(f"  Przetwarzanie pliku: {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("  BŁĄD: Nie można wczytać obrazu.")
        return None

    # 1. Konwersja do skali szarości, rozmycie i wykrywanie krawędzi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. Znalezienie konturów (kształtów) na obrazie
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  BŁĄD: Nie znaleziono żadnych konturów.")
        return None

    # Sortowanie konturów według pola powierzchni, od największych
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    for c in contours:
        # Aproksymacja konturu do prostszej figury
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Szukamy pierwszego z brzegu dużego konturu, który ma 4 rogi
        if len(approx) == 4 and cv2.contourArea(c) > 50000: # Minimalne pole, by odrzucić śmieci
            card_contour = approx
            break

    if card_contour is None:
        print("  BŁĄD: Nie udało się zidentyfikować karty na obrazie.")
        return None

    # 3. Transformacja perspektywy (prostowanie)
    rect = order_points(card_contour.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    # Obliczenie szerokości i wysokości docelowego obrazu
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Definiowanie docelowych rogów wyprostowanego obrazu
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Obliczenie macierzy transformacji i jej zastosowanie
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # 4. Poprawa kolorów (używając biblioteki Pillow)
    # Konwersja z formatu OpenCV (BGR) do Pillow (RGB)
    final_image_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    
    enhancer_br = ImageEnhance.Brightness(final_image_pil)
    final_image_pil = enhancer_br.enhance(WSP_JASNOSCI)

    enhancer_co = ImageEnhance.Contrast(final_image_pil)
    final_image_pil = enhancer_co.enhance(WSP_KONTRASTU)

    enhancer_sa = ImageEnhance.Color(final_image_pil)
    final_image_pil = enhancer_sa.enhance(WSP_NASYCENIA)

    print("  Przetwarzanie zakończone sukcesem.")
    return final_image_pil


def main():
    """Główna funkcja skryptu."""
    print("Inicjalizacja SANE...")
    sane.init()

    dzisiejsza_data = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(dzisiejsza_data):
        os.makedirs(dzisiejsza_data)
        print(f"Utworzono folder: {dzisiejsza_data}")

    skaner = None
    try:
        print(f"Otwieranie skanera: {NAZWA_SKANERA}")
        skaner = sane.open(NAZWA_SKANERA)

        print("Konfiguracja parametrów skanowania...")
        skaner.mode = 'Color'
        skaner.resolution = DPI
        
        # Skanujemy tylko jedną stronę (w pętli, gdybyś chciał skanować więcej)
        # Na razie pętla wykona się tylko raz.
        licznik = 1
        
        print("Połóż kartę na szybie skanera i naciśnij Enter, aby rozpocząć...")
        input() # Czekamy na użytkownika

        print(f"Skanowanie obrazu nr {licznik}...")
        temp_scan_path = "temp_scan.png"
        skaner.start()
        obraz_sane = skaner.snap()
        obraz_sane.save(temp_scan_path)
        print(f"Zapisano tymczasowy skan: {temp_scan_path}")

        # Przetwarzanie zeskanowanego obrazu
        finalny_obraz = process_image(temp_scan_path)
        
        if finalny_obraz:
            nazwa_pliku = os.path.join(dzisiejsza_data, f"karta_{licznik:03d}.png")
            finalny_obraz.save(nazwa_pliku)
            print(f"Zapisano finalny plik: {nazwa_pliku}")
        
        # Usunięcie tymczasowego pliku
        os.remove(temp_scan_path)

    except sane._sane.error as e:
        print(f"BŁĄD: Problem ze skanerem: {e}")
    finally:
        if skaner:
            skaner.close()
        sane.exit()
        print("Zakończono.")

if __name__ == "__main__":
    main()
