# 1. Analyse mit Training-Log
checker.py --repo https://github.com/foo/bar --log-training-data

# 2. Auto-Label (nutzt Second-Opinion Falls vorhanden)
python label.py --auto

# 3. Statistik anschauen
python label.py --stats

# 4. Manuell nachbessern
python label.py --interactive  # nur candidate-Einträge

# 5. Nur positive Einträge exportieren
python export_training_data.py --label positive --format sharegpt