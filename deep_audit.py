import os
import sys
import importlib.util
import traceback

print("=== ANTIGRAVITY PIPELINE DEEP AUDIT ===")
scripts_dir = os.path.join(os.getcwd(), 'scripts')
sys.path.insert(0, scripts_dir)

py_files = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
failed_imports = {}
passed = 0

for py_file in py_files:
    module_name = py_file[:-3]
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(scripts_dir, py_file))
        module = importlib.util.module_from_spec(spec)
        # Avoid running main blocks
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        passed += 1
    except Exception as e:
        failed_imports[py_file] = traceback.format_exc()

print(f"\n[1] Syntax & Compilation Audit: {passed}/{len(py_files)} files compiled successfully.")
if failed_imports:
    print("\n[!] Compilation Errors Found:")
    for f, err in failed_imports.items():
        print(f"  --> {f}")
        err_lines = err.strip().split('\n')[-3:]
        for line in err_lines:
            print(f"      {line}")

print("\n[2] Component Integrity Check")
import sqlite3
db_path = os.path.join(os.getcwd(), 'web', 'data', 'intelligence_hub.db')
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM matches")
    matches = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM training_examples")
    train_ex = cur.fetchone()[0]
    print(f"  [+] DB Intact: {matches} live matches, {train_ex} training examples.")
    conn.close()
else:
    print("  [-] DB not found.")

print("\n[3] Model Assets Check")
model_dir = os.path.join(os.getcwd(), 'model')
if os.path.exists(model_dir):
    model_count = sum(len(files) for _, _, files in os.walk(model_dir))
    size_mb = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(model_dir) for f in files) / (1024*1024)
    print(f"  [+] Found {model_count} neural assets ({size_mb:.2f} MB)")
else:
    print("  [-] No models found.")
