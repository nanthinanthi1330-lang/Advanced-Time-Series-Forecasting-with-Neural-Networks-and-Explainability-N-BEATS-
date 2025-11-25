
#!/bin/bash
python src/generate_data.py --hours 26280 --out data/sim.csv
python src/train_nbeats.py --config configs/nbeats_config.json
python src/run_baseline.py --csv data/sim.csv
python src/analyze.py
