import torch
import ultralytics

print("--- 1. RAPORT SYSTEMOWY ULTRALYTICS ---")
# To wypluje piękną tabelkę z wersją OS, Pythona i - co najważniejsze - wykrytym GPU
ultralytics.checks()

print("\n--- 2. TWARDE DANE Z PYTORCHA ---")
# Sprawdzamy stan faktyczny w samym silniku
cuda_dziala = torch.cuda.is_available()
print(f"Czy CUDA jest w ogóle dostępne dla PyTorcha? : {cuda_dziala}")

if cuda_dziala:
    liczba_gpu = torch.cuda.device_count()
    nazwa_gpu = torch.cuda.get_device_name(0)
    print(f"Liczba wykrytych kart graficznych         : {liczba_gpu}")
    print(f"Nazwa Twojego potwora (GPU 0)             : {nazwa_gpu}")
else:
    print("UWAGA: PyTorch nie widzi karty graficznej! Jedziemy na CPU.")
