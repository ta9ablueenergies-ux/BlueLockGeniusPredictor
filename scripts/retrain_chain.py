import subprocess, sys, os

leagues = ["LaLiga", "SerieA", "Bundesliga", "Ligue1"]
scripts_dir = os.path.dirname(os.path.abspath(__file__))

for league in leagues:
    log = os.path.join(scripts_dir, '..', f'retrain_{league.lower()}.log')
    log = os.path.normpath(log)
    print(f"[Chain] Starting {league}...", flush=True)
    with open(log, 'w') as out, open(log + '.err', 'w') as err:
        result = subprocess.run(
            [sys.executable, '-u', 'run_v11_hybrid.py', '--league', league, '--epochs', '60'],
            stdout=out, stderr=err, cwd=scripts_dir
        )
    print(f"[Chain] {league} done (exit={result.returncode})", flush=True)

print("[Chain] All leagues complete.", flush=True)
