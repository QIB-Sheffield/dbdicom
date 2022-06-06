import base64
import requests
url = 'https://api.github.com/repos/QIB-Sheffield/dbdicom/contents/README.md'
try:
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json()  # the response is a JSON
        content = base64.b64decode(req['content'])
        introduction = content.decode()
        __doc__ = introduction
except: # if there's no internet connection
    pass


# do not show in documentation
__pdoc__ = {}
__pdoc__["utilities"] = False 
__pdoc__["external"] = False 
__pdoc__["dicm"] = False 

from .folder import *
from .functions import *