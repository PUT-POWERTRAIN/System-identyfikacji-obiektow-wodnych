import argparse
import yaml
import os
import shutil
import ultralytics
from ultralytics import YOLO

def main():
    # --- 0. KONFIGURACJA ARGUMENTÓW Z LINII POLECEŃ ---
    parser = argparse.ArgumentParser(description="Skrypt do tuningu i treningu YOLOv11.")
    parser.add_argument('--data', type=str, required=True, 
                        help="Ścieżka do pliku datasetu data.yaml")
    parser.add_argument('--output', type=str, required=True, 
                        help="Katalog docelowy na gotowe wagi i wykresy (np. na Jetsona)")
    parser.add_argument('--project', type=str, default='./runs', 
                        help="Katalog roboczy dla YOLO (domyślnie: ./runs w obecnym folderze)")
    
    args = parser.parse_args()

    dataset_yaml = args.data
    destination_dir = args.output
    project_dir = args.project

    print("--- Inicjalizacja środowiska ---")
    ultralytics.checks()

    # --- ETAP 1: TUNING ---
    print("\n⏳ ETAP 1: Rozpoczynam tuning (30 prób po 50 epok)...")
    model_tune = YOLO('yolo11n.pt')

    model_tune.tune(
        data=dataset_yaml,
        epochs=50,
        iterations=30,
        optimizer='SGD',
        imgsz=640,
        project=project_dir,
        name='tuning_jetson'
    )

    # Wczytujemy idealne parametry
    best_params_path = os.path.join(project_dir, 'tuning_jetson', 'best_hyperparameters.yaml')
    
    try:
        with open(best_params_path, 'r') as f:
            best_params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Błąd krytyczny: Nie znaleziono pliku z parametrami {best_params_path}. Tuning mógł się nie udać.")
        return

    print("✅ ETAP 1 Zakończony! Przechodzę do Głównego Treningu.")

    # --- ETAP 2: WŁAŚCIWY TRENING ---
    print("\n🚀 ETAP 2: Rozpoczynam Właściwy Trening!")
    model_final = YOLO('yolo11n.pt')

    # Odpalamy trening z limitem 500 epok i Early Stopping
    results = model_final.train(
        data=dataset_yaml,
        epochs=500,       
        patience=50,      
        imgsz=640,
        project=project_dir,
        name='jetson_final_model',
        **best_params     # Ładuje parametry z tuningu
    )

    print("🎉 TRENING W PEŁNI ZAKOŃCZONY!")

    # --- ETAP 3: KOPIOWANIE PLIKÓW ---
    run_dir = os.path.join(project_dir, 'jetson_final_model')
    source_weights = os.path.join(run_dir, 'weights', 'best.pt')
    source_results_img = os.path.join(run_dir, 'results.png')
    source_results_csv = os.path.join(run_dir, 'results.csv')

    os.makedirs(destination_dir, exist_ok=True)

    print(f"\n📦 Rozpoczynam kopiowanie plików do katalogu docelowego: {destination_dir}")

    files_to_copy = {
        'Wagi modelu': (source_weights, 'yolo_jetson_best.pt'),
        'Wykresy krzywych': (source_results_img, 'krzywe_uczenia.png'),
        'Dane z epok (CSV)': (source_results_csv, 'historia_treningu.csv')
    }

    for name, (src, dest_name) in files_to_copy.items():
        dest = os.path.join(destination_dir, dest_name)
        try:
            shutil.copy(src, dest)
            print(f"✅ {name} skopiowane pomyślnie!")
        except FileNotFoundError:
            print(f"⚠️ Nie znaleziono: {src}. Trening prawdopodobnie nie doszedł do etapu zapisu.")
        except Exception as e:
            print(f"❌ Błąd kopiowania {name}: {e}")

    # --- ETAP 4: PODSUMOWANIE ---
    dest_results_img = os.path.join(destination_dir, 'krzywe_uczenia.png')
    print(f"\n🎉 Proces zakończony. Twoje pliki gotowe na Jetsona czekają w:\n{destination_dir}")
    
    if os.path.exists(dest_results_img):
        print(f"📊 Wykres krzywych uczenia został zapisany jako '{dest_results_img}'. Możesz go otworzyć w dowolnej przeglądarce zdjęć.")

if __name__ == "__main__":
    main()
