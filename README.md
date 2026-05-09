# System Identyfikacji Obiektów Wodnych

Repozytorium zawiera skrypty do treningu modeli YOLOv11 oraz inferencji z wykorzystaniem detekcji obiektów połączonej z estymacją głębi (Depth Anything V2).

## Wymagania i Instalacja

Aby uruchomić skrypty, konieczne jest zainstalowanie odpowiednich bibliotek. Przygotowany został skrypt `setup.sh`, który automatycznie instaluje pakiety z plików `requirements.txt` w głównym katalogu oraz submodułach.

Wymagania:
- Python 3.8+
- `pip`

Aby zainstalować zależności, wykonaj:

```bash
chmod +x setup.sh
./setup.sh
```

## Trening Modelu (trenowanie.py)

W repozytorium znajduje się skrypt `trenowanie.py`, który służy do pełnego procesu trenowania modelu YOLO z wbudowanym tuningiem hiperparametrów i mechanizmem Early Stopping. Skrypt jest zoptymalizowany m.in. z myślą o urządzeniach krawędziowych, np. Jetson.

### Jak uruchomić skrypt?

Wykorzystaj poniższą komendę:

```bash
python trenowanie.py --data <ścieżka_do_pliku_data.yaml> --output <katalog_docelowy> [--project <katalog_roboczy>]
```

### Argumenty:
- `--data`: (Wymagany) Ścieżka do pliku datasetu `data.yaml`.
- `--output`: (Wymagany) Katalog docelowy na gotowe wagi (np. dla Jetsona) i wykresy. Skrypt utworzy pliki w tym katalogu po udanym zakończeniu treningu.
- `--project`: (Opcjonalny) Katalog roboczy, w którym zapisywane są postępy Ultralytics. Domyślnie używa folderu `./runs`.

### Przebieg Działania:
1. **Tuning**: Skrypt wykona 30 prób wyszukiwania optymalnych hiperparametrów przez 50 epok używając SGD.
2. **Trening Główny**: Odbędzie się trening na 500 epok, z Early Stopping ustawionym na 50 epok bez poprawy, korzystając z odnalezionych hiperparametrów.
3. **Eksport**: Najlepsze wagi z treningu głównego oraz wykresy zostaną skopiowane do folderu wskazanego przez argument `--output` i gotowe do użycia na docelowym sprzęcie.

### Przykład:

```bash
python trenowanie.py --data /home/powertrain1/YOLO/datasets/merged_dataset_augmented/data.yaml --output ./wyniki_treningu
```
