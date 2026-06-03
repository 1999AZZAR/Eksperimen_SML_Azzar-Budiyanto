import subprocess, os
os.chdir(os.path.dirname(__file__) or '.')
cmd = [
    'venv/bin/jupyter', 'nbconvert',
    '--to', 'notebook',
    '--execute',
    '--ExecutePreprocessor.timeout=600',
    '--output', 'Eksperimen_Azzar-Budiyanto.ipynb',
    'Eksperimen_Azzar-Budiyanto.ipynb'
]
print('Executing notebook...')
result = subprocess.run(cmd, capture_output=False, text=True)
print(f'Exit code: {result.returncode}')
