#!/bin/bash

echo "Rozpoczynam instalację zależności dla projektu..."

# Aktualizacja pip
pip install --upgrade pip

# Instalacja podstawowych zależności z requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Instalacja pakietów z requirements.txt..."
    pip install -r requirements.txt
else
    echo "Brak pliku requirements.txt w głównym folderze!"
fi

# Instalacja zależności dla submodułu depth_and_distance_measure (jeśli istnieje)
if [ -f "depth_and_distance_measure/requirements.txt" ]; then
    echo "Instalacja dodatkowych zależności dla depth_anything_v2..."
    pip install -r depth_and_distance_measure/requirements.txt
fi

echo "Wszystkie zależności zostały zainstalowane pomyślnie!"
