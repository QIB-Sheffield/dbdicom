import os
# Use README.md for description in Github Page
with open(os.path.join(".", 'README.md'), encoding='utf-8') as f:
    introduction = f.read()
__doc__ = introduction

# do not show in documentation
__pdoc__ = {}
__pdoc__["utilities"] = False 
__pdoc__["external"] = False 
__pdoc__["dicm"] = False 

from .folder import *