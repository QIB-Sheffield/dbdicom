"""
============================
Maximum intensity projection
============================

Creating a maximum intensity projection of an image volume.
"""

import numpy as np
import dbdicom as db
from dbdicom.extensions.numpy import maximum_intensity_projection

# %%
# Create some test data, in this case a zero-filled array, describing 8 MRI images each measured at 3 flip angles and 2 repetition times:

coords = {
    'SliceLocation': np.arange(8),
    'FlipAngle': np.array([2, 15, 30]),
    'RepetitionTime': np.array([2.5, 5.0]),
}
series = db.zeros((128,128,8,3,2), gridcoords=coords)

# %%
# Create a maximum intensity projection

mip = maximum_intensity_projection(series, dims=tuple(coords), axis=0)

# %%
# To see what happened we can retrieve the nympy array of the MIP

array = mip.pixel_values(dims=('SliceLocation', 'ImageNumber'))
print(array.shape)


# sphinx_gallery_thumbnail_path = '_static/dbd.png'
