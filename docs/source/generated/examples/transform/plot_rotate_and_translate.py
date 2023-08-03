"""
====================
Rotate and translate
====================

Rotating and translating a 3D volume.

This example illustrates the use of the function `.rotate()` from the dbdicom extension :ref:`extension-vreg`. We use an ellipsoid digital reference object as test data and `.plot_surface()` from extension :ref:`extension-matplotlib` to visualise the results.
"""
import numpy as np

from dbdicom.dro import double_ellipsoid
from dbdicom.extensions.vreg import rotate
from dbdicom.extensions.matplotlib import plot_surface, plot_mosaic

# Choose image 1 as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = 2

# %%
# Generate and display an ellipsoid test object

ellipsoid_orig = double_ellipsoid(12, 40, 32, spacing=(2,3,1), levelset=True)
plot_surface(ellipsoid_orig)

# %%
# Define a rotation vector, apply it and display the result again.

# Define an anticlockwise rotation of 30 degrees around the y-axis
rotation = -30*(np.pi/180)*np.array([0,1,0])

# Perform the rotation and return a rotated series
ellipsoid_rot = rotate(ellipsoid_orig, rotation, reshape=True, mode='nearest')

# Display the surface of the rotated shape
plot_surface(ellipsoid_rot)

# %%
# Display the rotated shape as a mosaic

plot_mosaic(ellipsoid_rot) 

# %%
# When applying the rotation we used the `mode='nearest'` so that the values outside the boundaries of the volume are filled by nearest neighbour sampling. The default setting would fill these with a `constant=0` value, producing an additional surface at the edge of the volume:

ellipsoid_rot = rotate(ellipsoid_orig, rotation, reshape=True)
plot_surface(ellipsoid_rot)


# %%
# We used `reshape=True` so the new volume would encompass the entire shape. Running this with the default setting of `reshape=False` retains the original image shape and therefore misses part of the rotated volume:

ellipsoid_rot = rotate(ellipsoid_orig, rotation, reshape=False, mode='nearest')
plot_surface(ellipsoid_rot)