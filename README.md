# skanowanie_linux

## Wymagane zależności

Do uruchomienia skryptu potrzebne są następujące pakiety Pythona:

- `opencv-python-headless` (moduł `cv2`)
- `numpy`
- `pillow`

Możesz je zainstalować poleceniem:

```bash
pip install opencv-python-headless numpy pillow
```

Jeśli brakuje któregoś z modułów, skrypt zakończy się z czytelną informacją o brakującej zależności oraz przykładową komendą instalacji.

## Dostrajanie jasności i kontrastu

Domyślne wartości wzmacniające obraz znajdziesz w sekcji konfiguracji pliku `scan.py`:

```python
WSP_JASNOSCI = 1.20
WSP_KONTRASTU = 1.30
WSP_NASYCENIA = 1.10
```

Jeśli Twoje skany wychodzą zbyt jasne lub zbyt ciemne, możesz dopasować powyższe współczynniki – mniejsze wartości (np. `1.05`) łagodzą efekt, a większe (np. `1.40`) dodatkowo go wzmacniają. Po każdej zmianie uruchom próbne przetwarzanie, aby ocenić rezultat i dobrać parametry pod własny skaner.
