# 🌊 System Identyfikacji Obiektów Wodnych

![Vision](https://img.shields.io/badge/Vision-AI-blue) ![YOLOv11](https://img.shields.io/badge/YOLO-v11-green) ![DepthAnythingV2](https://img.shields.io/badge/Depth-Anything-orange)

Zaawansowany system do detekcji i analizy obiektów wodnych przy użyciu najnowszych modeli AI. System łączy w sobie precyzyjną detekcję obiektów (YOLO) z estymacją głębi w czasie rzeczywistym oraz wizualizacją 3D.

---

## 🛠️ Funkcjonalności

- **Detekcja 2D (YOLO)**: Szybka i precyzyjna identyfikacja łodzi, boj i przeszkód.
- **Estymacja Głębi (Depth)**: Obliczanie odległości do wykrytych obiektów bez użycia kamery stereo.
- **Wizualizacja 3D (Open3D)**: Przeniesienie obrazu z kamery w przestrzeń 3D wraz z chmurą punktów.
- **BEV (Bird's Eye View)**: Widok z lotu ptaka ułatwiający orientację w terenie.

---

## 🚀 Jak uruchomić i obsługiwać?

Wszystkie skrypty zostały zaktualizowane, aby przyjmować argumenty z linii poleceń. Poniżej znajdziesz szczegóły dotyczące każdego z nich oraz instrukcje interakcji.

### 1. 🎥 Podgląd 3D (Pełny system)
Najbardziej zaawansowany skrypt łączący detekcję, głębię i chmurę punktów.

```bash
python 3d_camera_feed.py --model /home/powertrain1/YOLO/models/yolo_jetson_best.engine --camera 0
```

**Co zobaczysz:**
- **Okno "3D Detection"**: Trzy panele (Obraz z kamery z ramkami, Mapa głębi, Widok BEV - z lotu ptaka).
- **Okno "3D Visualization"**: Interaktywna chmura punktów 3D generowana w czasie rzeczywistym.

**Kluczowe argumenty:**
- `--model`: Ścieżka do modelu YOLO (zalecany `.engine` dla Jetsona).
- `--camera`: Indeks kamery (np. `0`) lub ścieżka do pliku `.mp4`.
- `--conf`: Próg pewności detekcji (domyślnie `0.85`).
- `--imgsz`: Rozmiar obrazu (domyślnie `640`). **Ważne dla modeli .engine!**

**Sterowanie:**
- `q`: Zamyka oba okna i kończy działanie skryptu.
- **Myszka (w oknie 3D)**: Lewy przycisk - obracanie, Prawy przycisk - przesuwanie, Scroll - zoom.

---

### 2. ⚡ Szybki Podgląd 2D (Tylko YOLO)
Lekki skrypt, gdy liczy się tylko prędkość detekcji.

```bash
python 2d_camera_feed.py --model /home/powertrain1/YOLO/models/yolo_jetson_best.engine --camera 0
```

**Co zobaczysz:**
- Pojedyncze okno z obrazem z kamery, ramkami detekcji oraz licznikiem **FPS** w lewym górnym rogu.

**Kluczowe argumenty:**
- `--model`: Ścieżka do modelu YOLO.
- `--camera`: Indeks kamery lub ścieżka wideo.
- `--conf`: Próg pewności.
- `--imgsz`: Rozmiar obrazu (musi zgadzać się z eksportem `.engine`).

**Sterowanie:**
- `q`: Wyjście ze skryptu.

---

### 3. 🖼️ Analiza Pojedynczego Zdjęcia
Przetwarzanie statycznych plików dla dokładnej analizy.

```bash
python yolo_script.py --image test_images/images2.jpeg --model /home/powertrain1/YOLO/models/lodz.pt
```

**Kluczowe argumenty:**
- `--image`: Ścieżka do zdjęcia wejściowego.
- `--model`: Model YOLO (może być `.pt` lub `.engine`).
- `--output`: Katalog, gdzie zostanie zapisany wynik (domyślnie `results/`).
- `--imgsz`: Rozmiar obrazu dla modelu.

**Wynik:**
- Skrypt zapisuje przetworzony obraz w folderze `results/`. Nazwa pliku zaczyna się od `result_`.

---

### 4. 🎓 Trening Modelu
Pipeline dla twórców modeli.

```bash
python trenowanie.py --data data.yaml --output exports/ --model yolo11n.pt
```

**Kluczowe argumenty:**
- `--data`: Ścieżka do pliku konfiguracyjnego datasetu `data.yaml`.
- `--output`: Gdzie zapisać gotowe wagi i wykresy po treningu.
- `--model`: Bazowy model YOLO (np. `yolo11n.pt` lub `yolo11s.pt`).
- `--project`: Katalog roboczy dla logów YOLO.

---

## ⚙️ Ważne: Modele .engine i Jetson

Modele w formacie **.engine** są zoptymalizowane pod procesory NVIDIA (TensorRT) i oferują najwyższą wydajność na Jetsonie. 

> [!WARNING]
> Modele `.engine` mają **na sztywno zdefiniowany rozmiar obrazu** (imgsz) oraz **wielkość paczki** (batch size). Jeśli model został wyeksportowany dla `imgsz=640`, musisz użyć argumentu `--imgsz 640` przy uruchamianiu skryptu, inaczej skrypt może się zawiesić lub działać bardzo wolno.

---

## 🛠️ Argumenty wspólne (Opcjonalne)
- `--conf`: Próg pewności (0.0 - 1.0). Domyślnie `0.85`.
- `--imgsz`: Rozmiar obrazu wejściowego. Domyślnie `640`.
- `--depth_weights`: Ścieżka do wag modelu głębi (domyślnie szuka w głównym folderze).


---

## 📂 Struktura Projektu

- `3d_camera_feed.py`: Główny skrypt z wizualizacją 3D i estymacją odległości.
- `2d_camera_feed.py`: Lekka wersja tylko z detekcją YOLO.
- `yolo_script.py`: Przetwarzanie pojedynczych zdjęć.
- `trenowanie.py`: Pipeline treningowy (Tuning -> Trening -> Export).
- `depth_and_distance_measure/`: Moduły do estymacji głębi.

---

## ⚠️ Wymagania i Instalacja

Zalecane środowisko: **NVIDIA Jetson** lub PC z kartą **CUDA**.

Aby przygotować środowisko, możesz użyć gotowego skryptu instalacyjnego, który zainstaluje wszystkie zależności:

```bash
chmod +x setup.sh
./setup.sh
```

Alternatywnie, możesz zainstalować biblioteki ręcznie przez pip:

```bash
pip install -r requirements.txt
```

---
*Powodzenia, żołnierzu! 🫡*
