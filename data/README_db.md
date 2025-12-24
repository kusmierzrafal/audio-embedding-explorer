### create database (windows powershell)

python .\init_db.py
python .\check_db.py
python .\run_sql.py .\seed_models.sql
python .\ingest_audio.py .\audio_seed --recursive
python .\verify_db.py
