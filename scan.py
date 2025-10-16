import os
import sys
import importlib.util
import shutil
from datetime import datetime

import sane


def ensure_module(name: str, install_hint: str) -> None:
    """Sprawdza dostępność modułu i kończy program z czytelnym komunikatem."""
    if importlib.util.find_spec(name) is None:
        print(
            f"Brak modułu '{name}'. Zainstaluj go poleceniem: {install_hint}",
            file=sys.stderr,
        )
        sys.exit(1)


ensure_module("cv2", "pip install opencv-python-headless")
ensure_module("numpy", "pip install numpy")
ensure_module("PIL", "pip install pillow")

import cv2  # OpenCV do analizy obrazu
import numpy as np
from PIL import Image, ImageEnhance

# --- Konfiguracja ---
DPI = 600
CARD_WIDTH_MM = 63.5
CARD_HEIGHT_MM = 88.9
DIMENSION_TOLERANCE = 0.10  # 10%

# Nazwa skanera pozostała dla kompatybilności, ale nie jest używana bezpośrednio.
NAZWA_SKANERA = "fujitsu:fi-630dj:13583"  # Domyślna nazwa skanera (fallback)

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

    mm_to_px = DPI / 25.4
    expected_short_side_px = min(CARD_WIDTH_MM, CARD_HEIGHT_MM) * mm_to_px
    expected_long_side_px = max(CARD_WIDTH_MM, CARD_HEIGHT_MM) * mm_to_px
    expected_ratio = expected_long_side_px / expected_short_side_px

    short_min = expected_short_side_px * (1 - DIMENSION_TOLERANCE)
    short_max = expected_short_side_px * (1 + DIMENSION_TOLERANCE)
    long_min = expected_long_side_px * (1 - DIMENSION_TOLERANCE)
    long_max = expected_long_side_px * (1 + DIMENSION_TOLERANCE)
    ratio_min = expected_ratio * (1 - DIMENSION_TOLERANCE)
    ratio_max = expected_ratio * (1 + DIMENSION_TOLERANCE)

    print(
        "  Oczekiwane wymiary karty (px):"
        f" krótszy bok ~{expected_short_side_px:.1f} ({short_min:.1f}-{short_max:.1f}),"
        f" dłuższy bok ~{expected_long_side_px:.1f} ({long_min:.1f}-{long_max:.1f}),"
        f" stosunek boków ~{expected_ratio:.3f} ({ratio_min:.3f}-{ratio_max:.3f})"
    )

    # Sortowanie konturów według pola powierzchni, od największych
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    for c in contours:
        # Aproksymacja konturu do prostszej figury i podstawowe filtrowanie
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if area <= 3000:  # Minimalne pole, by odrzucić śmieci
            continue

        rect = cv2.minAreaRect(c)
        (width_px, height_px) = rect[1]
        if width_px == 0 or height_px == 0:
            continue

        short_side, long_side = sorted((width_px, height_px))
        ratio = long_side / short_side
        approx_area = width_px * height_px

        short_ok = short_min <= short_side <= short_max
        long_ok = long_min <= long_side <= long_max
        ratio_ok = ratio_min <= ratio <= ratio_max

        print(
            "    Kontur:"
            f" krótki bok={short_side:.1f}px (ok={short_ok}),"
            f" długi bok={long_side:.1f}px (ok={long_ok}),"
            f" ratio={ratio:.3f} (ok={ratio_ok}),"
            f" pole~{approx_area:.0f}px^2"
        )

        if not (short_ok and long_ok and ratio_ok):
            continue

        # Jeśli aproksymowany kontur ma 4 rogi – wykorzystujemy je
        if len(approx) == 4:
            card_contour = approx.reshape(4, 2).astype("float32")
        else:
            box = cv2.boxPoints(rect)
            card_contour = np.array(box, dtype="float32")
        print("    -> Kontur zaakceptowany jako karta.")
        break

    if card_contour is None:
        print("  BŁĄD: Nie udało się zidentyfikować karty na obrazie.")
        return None

    # 3. Transformacja perspektywy (prostowanie)
    rect = order_points(card_contour)
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

    devices = sane.get_devices() or []
    if devices:
        print("Wykryte skanery:")
        for idx, device in enumerate(devices):
            nazwa, producent, model, typ = device
            opis = f"{producent} {model}".strip()
            typ_info = f" ({typ})" if typ else ""
            print(f"  [{idx}] {nazwa} - {opis}{typ_info}")
    else:
        print("Nie wykryto żadnych skanerów przez SANE.")

    fujitsu_matches = [device for device in devices if device[2] and "fi-6130" in device[2].lower()]
    wybrane_urzadzenie = None

    if len(fujitsu_matches) == 1:
        wybrane_urzadzenie = fujitsu_matches[0]
        nazwa, producent, model, _ = wybrane_urzadzenie
        print(f"Automatycznie wybrano skaner: {producent} {model} ({nazwa})")
    elif devices:
        print("Wybierz skaner z listy wpisując numer odpowiadający urządzeniu (Enter = 0):")
        while True:
            wybor = input("> ").strip()
            if wybor == "":
                wybor = "0"
            try:
                indeks = int(wybor)
            except ValueError:
                print("Nieprawidłowy wybór. Spróbuj ponownie.")
                continue

            if 0 <= indeks < len(devices):
                wybrane_urzadzenie = devices[indeks]
                nazwa, producent, model, _ = wybrane_urzadzenie
                print(f"Wybrano skaner: {producent} {model} ({nazwa})")
                break

            print("Wybrany numer jest poza zakresem. Spróbuj ponownie.")

    if wybrane_urzadzenie is None:
        if devices:
            print("Nie udało się wybrać skanera. Używanie domyślnej konfiguracji.")
        else:
            print("Brak dostępnych skanerów. Zakończono.")
            sane.exit()
            return

    nazwa_skanera = wybrane_urzadzenie[0] if wybrane_urzadzenie else NAZWA_SKANERA

    dzisiejsza_data = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(dzisiejsza_data):
        os.makedirs(dzisiejsza_data)
        print(f"Utworzono folder: {dzisiejsza_data}")

    skaner = None
    try:
        print(f"Otwieranie skanera: {nazwa_skanera}")
        skaner = sane.open(nazwa_skanera)

        print("Konfiguracja parametrów skanowania...")
        skaner.mode = 'Color'
        skaner.resolution = DPI
        
        # Skanujemy kolejne karty do momentu przerwania pętli.
        licznik = 1

        while True:
            print("Połóż kartę na szybie skanera i naciśnij Enter, aby rozpocząć...")
            input()  # Czekamy na użytkownika

            print(f"Skanowanie obrazu nr {licznik}...")
            temp_scan_path = "temp_scan.png"

            try:
                skaner.start()
                obraz_sane = skaner.snap()
            except sane._sane.error as e:
                blad = str(e).lower()
                pusty_podajnik = any(
                    komunikat in blad
                    for komunikat in (
                        "document feeder out of documents",
                        "document feeder empty",
                        "feeder empty",
                        "no documents",
                    )
                )

                if pusty_podajnik:
                    decyzja = input(
                        "Podajnik pusty – [Enter] aby kontynuować, 'q' by zakończyć: "
                    ).strip().lower()
                    if decyzja == "q":
                        print("Przerwano skanowanie na życzenie użytkownika.")
                        break
                    print("Kontynuuję oczekiwanie na kolejną kartę.")
                    continue

                print(f"BŁĄD: Problem ze skanerem: {e}")
                break

            obraz_sane.save(temp_scan_path)
            print(f"Zapisano tymczasowy skan: {temp_scan_path}")

            oryginal_path = os.path.join(
                dzisiejsza_data, f"karta_{licznik:03d}_oryginal.png"
            )
            shutil.copy(temp_scan_path, oryginal_path)
            print(f"Zachowano oryginalny skan w: {oryginal_path}")

            # Przetwarzanie zeskanowanego obrazu
            finalny_obraz = process_image(temp_scan_path)

            if finalny_obraz is not None:
                nazwa_pliku = os.path.join(dzisiejsza_data, f"karta_{licznik:03d}.png")
                finalny_obraz.save(nazwa_pliku)
                print(f"Zapisano finalny plik: {nazwa_pliku}")
            else:
                print(
                    "  UWAGA: Nie udało się przetworzyć obrazu. Oryginalny skan dostępny w: "
                    f"{oryginal_path}"
                )
                if os.path.exists(temp_scan_path):
                    os.remove(temp_scan_path)

            licznik += 1

    except sane._sane.error as e:
        print(f"BŁĄD: Problem ze skanerem: {e}")
    finally:
        if skaner:
            skaner.close()
        sane.exit()
        print("Zakończono.")

if __name__ == "__main__":
    main()
