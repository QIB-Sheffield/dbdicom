"""
==================================
T1-mapping with variable TR and FA
==================================

5-dimensions dataset simulating MRI data collected fro T1-mapping using the technical of variable flip angles and repetition times. 
"""

# %%
# Gnerate synthetic data and extract coordinates and array.
import numpy as np
from dbdicom import dro

data = dro.T1_mapping_vFATR()
dims = ('SliceLocation', 'FlipAngle', 'RepetitionTime')
#coords = data.coords(dims)


# %%
# Display the data at the smallest flip angle and repetition time

from dbdicom.extensions.matplotlib import plot_mosaic

im0 = data.extract(FlipAngle=20, RepetitionTime=5.0)
plot_mosaic(im0, zdim='SliceLocation', gridspacing=500) 


# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
