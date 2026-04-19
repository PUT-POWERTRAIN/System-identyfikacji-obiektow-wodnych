"""
Skrypt augmentacji datasetu YOLO (buoy + dock).
Generuje 3 augmentowane kopie każdego zdjęcia treningowego.
Walidacja pozostaje nienaruszona.

Wejście:  merged_dataset/
Wyjście:  merged_dataset_augmented/
"""

import os
import cv2
import shutil
import random
import numpy as np
import albumentations as A
from pathlib import Path
import sys

# ======================== KONFIGURACJA ========================
SRC_DIR = Path("merged_dataset")
DST_DIR = Path("merged_dataset_augmented")
NUM_AUGMENTATIONS = 3  # ile augmentowanych kopii per obraz (3 + oryginał = 4x)
RANDOM_SEED = 42
# ==============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_augmentation_pipelines():
    """
    Zwraca listę pipeline'ów augmentacji.
    Każdy pipeline ma inny zestaw transformacji, żeby uzyskać różnorodność.
    """
    # Pipeline 1: Geometryczne transformacje
    aug1 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=0, p=0.5),
        A.RandomResizedCrop(size=(640, 640), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'],
        min_area=100, min_visibility=0.3
    ))

    # Pipeline 2: Kolorystyczne + pogodowe
    aug2 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=1),
            A.RandomRain(slant_range=(-10, 10), drop_length=10, drop_width=1,
                         drop_color=(200, 200, 200), blur_value=3, p=1),
            A.RandomSunFlare(src_radius=100, p=1),
        ], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'],
        min_area=100, min_visibility=0.3
    ))

    # Pipeline 3: Szum + blur + mieszane
    aug3 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        ], p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
    ], bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'],
        min_area=100, min_visibility=0.3
    ))

    return [aug1, aug2, aug3]


def read_yolo_labels(label_path):
    """Czyta plik labelek YOLO, zwraca listę (class_id, x, y, w, h)."""
    bboxes = []
    class_labels = []

    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Klampuj do [0, 1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))
                bboxes.append([x, y, w, h])
                class_labels.append(cls)

    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    """Zapisuje labelki YOLO do pliku."""
    with open(label_path, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            x, y, w, h = bbox
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def augment_image(img, bboxes, class_labels, augmentation):
    """Augmentuje obraz i bounding boxy."""
    try:
        result = augmentation(image=img, bboxes=bboxes, class_labels=class_labels)
        return result['image'], result['bboxes'], result['class_labels']
    except Exception as e:
        # Jeśli augmentacja nie powiodła się (np. bbox poza granicami), zwróć None
        print(f"  Augmentacja pominięta: {e}")
        return None, None, None


def copy_directory(src, dst):
    """Kopiuje katalog z plikami."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    print("=" * 60)
    print("  AUGMENTACJA DATASETU YOLO")
    print(f"  Źródło:  {SRC_DIR.resolve()}")
    print(f"  Cel:     {DST_DIR.resolve()}")
    print(f"  Kopie:   {NUM_AUGMENTATIONS} augmentacji per obraz + oryginał")
    print("=" * 60)

    # Sprawdź czy source istnieje
    if not SRC_DIR.exists():
        print(f"BŁĄD: Katalog źródłowy {SRC_DIR} nie istnieje!")
        return

    # Usuń stary output jeśli istnieje
    if DST_DIR.exists():
        print(f"\nUsuwam stary katalog {DST_DIR}...")
        shutil.rmtree(DST_DIR)

    # Tworzenie struktury wyjściowej
    dst_train_img = DST_DIR / "train" / "images"
    dst_train_lbl = DST_DIR / "train" / "labels"
    dst_val_img = DST_DIR / "val" / "images"
    dst_val_lbl = DST_DIR / "val" / "labels"

    dst_train_img.mkdir(parents=True, exist_ok=True)
    dst_train_lbl.mkdir(parents=True, exist_ok=True)

    # Kopiuj walidację bez zmian
    print("\n📋 Kopiuję zbiór walidacyjny (bez augmentacji)...")
    src_val_img = SRC_DIR / "val" / "images"
    src_val_lbl = SRC_DIR / "val" / "labels"
    if src_val_img.exists():
        copy_directory(src_val_img, dst_val_img)
    if src_val_lbl.exists():
        copy_directory(src_val_lbl, dst_val_lbl)
    val_count = len(list(dst_val_img.glob("*"))) if dst_val_img.exists() else 0
    print(f"   Skopiowano {val_count} obrazów walidacyjnych.")

    # Pobierz pipeline'y augmentacji
    augmentations = get_augmentation_pipelines()

    # Augmentuj zbiór treningowy
    src_train_img = SRC_DIR / "train" / "images"
    src_train_lbl = SRC_DIR / "train" / "labels"

    image_files = sorted([
        f for f in src_train_img.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ])

    print(f"\n🔧 Augmentuję zbiór treningowy ({len(image_files)} obrazów × {NUM_AUGMENTATIONS + 1})...")
    print(f"   Oczekiwana liczba obrazów: ~{len(image_files) * (NUM_AUGMENTATIONS + 1)}")

    original_count = 0
    augmented_count = 0
    skipped_count = 0

    for i, img_path in enumerate(image_files):
        if i % 50 == 0:
            print(f"  [{i}/{len(image_files)}] przetwarzam...", flush=True)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  UWAGA: Nie udało się wczytać {img_path.name}, pomijam.")
            skipped_count += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Wczytaj labelki
        label_file = src_train_lbl / (img_path.stem + ".txt")
        bboxes, class_labels = read_yolo_labels(str(label_file))

        # 1) Zapisz oryginał
        dst_img_path = dst_train_img / img_path.name
        cv2.imwrite(str(dst_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        dst_lbl_path = dst_train_lbl / (img_path.stem + ".txt")
        if bboxes:
            write_yolo_labels(str(dst_lbl_path), bboxes, class_labels)
        elif label_file.exists():
            shutil.copy(str(label_file), str(dst_lbl_path))
        original_count += 1

        # 2) Generuj augmentowane kopie
        for aug_idx in range(NUM_AUGMENTATIONS):
            aug_pipeline = augmentations[aug_idx % len(augmentations)]

            aug_img, aug_bboxes, aug_cls = augment_image(
                img, bboxes, class_labels, aug_pipeline
            )

            if aug_img is None:
                skipped_count += 1
                continue

            # Nazwa: original_aug1.jpg, original_aug2.jpg, ...
            aug_name = f"{img_path.stem}_aug{aug_idx + 1}{img_path.suffix}"
            aug_img_path = dst_train_img / aug_name
            aug_lbl_path = dst_train_lbl / f"{img_path.stem}_aug{aug_idx + 1}.txt"

            cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            if aug_bboxes:
                write_yolo_labels(str(aug_lbl_path), aug_bboxes, aug_cls)

            augmented_count += 1

    # Generuj data.yaml
    yaml_content = f"""path: {DST_DIR.resolve()}
train: train/images
val: val/images

nc: 2
names: ['buoy', 'dock']
"""
    yaml_path = DST_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Podsumowanie
    total_train = original_count + augmented_count
    print("\n" + "=" * 60)
    print("  ✅ AUGMENTACJA ZAKOŃCZONA!")
    print("=" * 60)
    print(f"  Oryginalne obrazy treningowe:  {original_count}")
    print(f"  Augmentowane obrazy:           {augmented_count}")
    print(f"  Łącznie treningowych:          {total_train}")
    print(f"  Obrazy walidacyjne:            {val_count}")
    print(f"  Pominięte:                     {skipped_count}")
    print(f"\n  Mnożnik:                       {total_train / max(original_count, 1):.1f}x")
    print(f"\n  Dataset YAML:  {yaml_path.resolve()}")
    print(f"  Output dir:    {DST_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
