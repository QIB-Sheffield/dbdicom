"""Generate dbdicom documentation"""

import os, sys

# COMMAND LINE SCRIPT
# py -3 -m venv .venv
# .venv\\scripts\\activate
# py -m pip install -r requirements.txt
# pdoc --html -f -c sort_identifiers=False weasel  

windows = (sys.platform == "win32") or (sys.platform == "win64") or (os.name == 'nt')
if windows:
    activate = '.venv\\scripts\\activate'
else: # MacOS and Linux
    activate = '.venv/bin/activate' 

print('Creating virtual environment..')
os.system('py -3 -m venv .venv')

print('Installing requirements..')
os.system(activate + ' && ' + 'py -m pip install -r requirements.txt')

print('Generating documentation..')
os.system(activate + ' && ' + 'pdoc --html -f -c sort_identifiers=False dbdicom')