import torch
from ultralytics import YOLO

print("--- 1. DIAGNOSTYKA SPRZĘTU ---")
# Sprawdzamy, czy mamy akcelerację
if torch.cuda.is_available():
    nazwa_gpu = torch.cuda.get_device_name(0)
    print(f"Status: SUKCES! Model za chwilę wleci na: {nazwa_gpu}")
else:
    print("Status: UWAGA! Brak CUDA, lecimy na powolnym CPU!")

print("\n--- 2. ŁADOWANIE MODELU ---")
model = YOLO('yolov8n.pt')

print("\n--- 3. ROZGRZEWKA KARTY (Warm-up) ---")
# Pierwsze przepuszczenie zdjęcia (verbose=False żeby nie śmiecić w konsoli)
model('bus.jpg', device='cuda')
print("Karta rozgrzana, rdzenie gotowe.")

print("\n--- 4. WŁAŚCIWA DETEKCJA ---")
# Prawdziwy test prędkości Jetsona
results = model('bus.jpg', device='cuda')

# Zapis i koniec
for r in results:
    r.save(filename='wynik.jpg')

print("\nGotowe mordo! Wynik z ramkami zapisany jako 'wynik.jpg'.")
