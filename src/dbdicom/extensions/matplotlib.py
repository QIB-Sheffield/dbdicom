import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from skimage import measure

from dbdicom.utils.image import as_mosaic


def plot_mosaic(series, zdim='InstanceNumber', rows:int=None, colorbar=True, clim:tuple=None, gridspacing=100, size=10):
    """Show a 3D array as a mosaic

    Args:
        zdim (str, optional): slice dimension. Defaults to 'InstanceNumber'
        series (dbdicom.Series): Series with the array to be plotted.
        rows (int, optional): Number of rows of the mosaic. If set to None the mosaic will be chosen to be approximately square. Defaults to None.
        colorbar (bool, optional): If True, a color bar is shown next to the mosaic. Defaults to True.
        clim (tuple of 2 elements, optional): if  provided, this determines the minimal and maximal signal values shown. If it is set to None or not provided, it will be taken from the WindowCenter and WindowWidth values in the DICOM header. Defaults to None.
        gridspacing (int, optional): spacing in mm between gridlines on the plot. Defaults to 100 mm.
        size (float, optional): size of the largest dimension of the image in inches. Defaults to 10 inch
    """
    array = series.pixel_values(dims=(zdim,))
    spacing = series.spacing()
    mosaic = as_mosaic(array, rows=rows)
    cols = mosaic.shape[1]/spacing[1]

    if clim is None:
        c, w = series.WindowCenter, series.WindowWidth
        clim = (c-w/2, c+w/2)
    fig, ax = plt.subplots()
    dx, dy = spacing[0]*mosaic.shape[0], spacing[1]*mosaic.shape[1]
    size = 10
    fig.set_size_inches(size*dx/np.amax([dx,dy]), size*dy/np.amax([dx,dy]))
    im = ax.imshow(mosaic.T, clim=clim)
    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("y-axis (mm)")
    x = np.arange(0, 1+mosaic.shape[0], gridspacing/spacing[0])
    y = np.arange(0, 1+mosaic.shape[1], gridspacing/spacing[1])
    ax.set_xticks(x, np.int16(x*spacing[0]))
    ax.set_yticks(y, np.int16(y*spacing[1]))
    if colorbar:
        fig.colorbar(im, ax=ax, label='Signal')
    plt.show()


def plot_surface(series, level=0, gridspacing=10, show=True):
    """Plot the surface at a given level

    Args:
        series (dbdicom.Series): Series with the array to be plotted.
        level (int, optional): Extract the surface at this level. Defaults to 0.
        gridspacing (int, optional): spacing in mm between gridlines on the plot. Defaults to 10 mm.
        show (bool, optional): if True, the function displays the plot. Defaults to True.

    Returns:
        Figure: matplotlib figure with the plot.

    Requires:
        matplotlib
        skimage
    """

    # Extract the numpy array from the dbdicom series:
    array = series.pixel_values()
    spacing = series.spacing()
    #spacing = (spacing[1], spacing[0], spacing[2])

    # Use marching cubes to extract the surface from the array
    verts, faces, _, _ = measure.marching_cubes(array, level)

    # Use the matplotlib toolkit to create a trinagular mesh
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("y-axis (mm)")
    ax.set_zlabel("z-axis (mm)")

    x = np.arange(0, 1+array.shape[0], gridspacing/spacing[0])
    y = np.arange(0, 1+array.shape[1], gridspacing/spacing[1])
    z = np.arange(0, 1+array.shape[2], gridspacing/spacing[2])

    ax.set_xticks(x, np.int16(x*spacing[0]))
    ax.set_yticks(y, np.int16(y*spacing[1]))
    ax.set_zticks(z, np.int16(z*spacing[2]))

    # Hack to ensure proportional size of each axis
    x_len = array.shape[0]*spacing[0]
    y_len = array.shape[1]*spacing[1]
    z_len = array.shape[2]*spacing[2]

    scale=np.diag([x_len, y_len, z_len, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj

    if show:
        plt.show()