import os
import subprocess

# Paths for clean and unclean datasets
clean_file = os.path.join("examples", "clean_data.csv")
unclean_file = os.path.join("examples", "unclean_data.csv")

# Run the package via CLI
subprocess.run(["python", "main.py", clean_file, unclean_file])

print("Example execution complete. Check 'outputs/' for results.")
