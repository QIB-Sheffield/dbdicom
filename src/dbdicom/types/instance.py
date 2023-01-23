import timeit
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt


from dbdicom.record import DbRecord
from dbdicom.ds.create import new_dataset
import dbdicom.utils.image as image


class Instance(DbRecord):

    name = 'SOPInstanceUID'

    def keys(self):
        return [self.key()]

    def parent(self):
        #uid = self.manager.register.at[self.key(), 'SeriesInstanceUID']
        uid = self.manager._at(self.key(), 'SeriesInstanceUID')
        return self.record('Series', uid, key=self.key())

    def children(self, **kwargs):
        return

    def new_child(self, **kwargs): 
        return

    def _copy_from(self, record, **kwargs):
        return

    def copy_to_series(self, series):
        uid = self.manager.copy_instance_to_series(self.key(), series.keys(), series)
        return self.record('Instance', uid)

    def array(self):
        return self.get_pixel_array()

    def get_pixel_array(self):
        ds = self.get_dataset()
        return ds.get_pixel_array()

    def set_array(self, array):
        self.set_pixel_array(array)
        
    def set_pixel_array(self, array):
        ds = self.get_dataset()
        if ds is None:
            ds = new_dataset('MRImage')
        ds.set_pixel_array(array)
        self.set_dataset(ds)

    def set_dataset(self, dataset):
        self._key = self.manager.set_instance_dataset(self.uid, dataset, self.key())

    def map_to(self, target):
        return map_to(self, target)

    def map_mask_to(self, target):
        return map_mask_to(self, target)

    def export_as_csv(*args, **kwargs):
        export_as_csv(*args, **kwargs)

    def export_as_png(*args, **kwargs):
        export_as_png(*args, **kwargs)

    def export_as_nifti(*args, **kwargs):
        export_as_nifti(*args, **kwargs)

    def BGRA_array(self):
        return image.BGRA(
            self.get_pixel_array(),
            self.lut, 
            width = self.WindowWidth,
            center = self.WindowCenter,
        )


def map_to(source, target):
    """Map non-zero image pixels onto a target image.
    
    Overwrite pixel values in the target"""

    dss = source.get_dataset()
    dst = target.get_dataset()

    # Create a coordinate array for all pixels in the source
    coords = np.empty((dss.Rows*dss.Columns, 3), dtype=np.uint16)
    for x in range(dss.Columns):
        for y in range(dss.Rows):
            coords[x*dss.Columns+y,:] = [x,y,0]

    # Apply coordinate transformation from source to target
    affineSource = dss.get_affine_matrix()
    affineTarget = dst.get_affine_matrix()
    sourceToTarget = np.linalg.inv(affineTarget).dot(affineSource)
    coords_target = nib.affines.apply_affine(sourceToTarget, coords)

    # Interpolate (nearest neighbour) and extract inslice coordinates
    coords_target = np.round(coords_target, 3).astype(int)
    xt = tuple([c[0] for c in coords_target if c[2] == 0])
    yt = tuple([c[1] for c in coords_target if c[2] == 0])
    xs = tuple([c[0] for c in coords])
    ys = tuple([c[1] for c in coords])

    ## COORDINATES DO NOT MATCH UP because of c[2] = 0 condition
    ## Needs a different indexing approach

    # Set values in the target image
    source_array = dss.get_pixel_array()
    target_array = np.zeros((dst.Columns, dst.Rows))
    target_array[(xt, yt)] = source_array[(xs, ys)]
    # for masking map values to {0, 1}
    result = source.new_sibling()
    result.set_pixel_array(target_array)

    return result


def map_mask_to(record, target):
    """Map non-zero image pixels onto a target image.
    Overwrite pixel values in the target"""
    dsr = record.get_dataset()
    dst = target.get_dataset()
    array = dsr.map_mask_to(dst)
    result = target.copy_to(record.parent()) # inherit geometry header from target
    result.set_pixel_array(array)
    return result


def export_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export 2D pixel Array in csv format"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription

    filename = os.path.join(directory, filename + '.csv')
    table = record.get_pixel_array()
    if columnHeaders is None:
        columnHeaders = []
        counter = 0
        for _ in table:
            counter += 1
            columnHeaders.append("Column" + str(counter))
    df = pd.DataFrame(np.transpose(table), columns=columnHeaders)
    df.to_csv(filename, index=False)


def export_as_png(record, directory=None, filename=None):
    """Export image in png format."""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the png data')

    pixelArray = np.transpose(record.get_pixel_array())
    centre, width = record.window
    minValue = centre - width/2
    maxValue = centre + width/2
    #cmap = plt.get_cmap(colourTable)
    cmap = record.colormap
    if cmap is None:
        cmap='gray'
    #plt.imshow(pixelArray, cmap=cmap)
    plt.imshow(pixelArray, cmap=cmap, vmin=minValue, vmax=maxValue)
    #plt.imshow(pixelArray, cmap=colourTable)
    #plt.clim(int(minValue), int(maxValue))
    cBar = plt.colorbar()
    cBar.minorticks_on()
    if filename is None:
        filename = record.label()
    filename = os.path.join(directory, filename + '.png')
    plt.savefig(fname=filename + '.png')
    plt.close() 

def export_as_nifti(record, directory=None, filename=None):
    """Export series as a single Nifty file"""

    if directory is None: 
        directory = record.dialog.directory(message='Please select a folder for the nifty data')
    if filename is None:
        filename = record.SeriesDescription

    ds = record.get_dataset()
    dicomHeader = nib.nifti1.Nifti1DicomExtension(2, ds)
    array = record.get_pixel_array()
    niftiObj = nib.Nifti1Image(array, ds.get_values('affine_matrix'))
    niftiObj.header.extensions.append(dicomHeader)
    nib.save(niftiObj, directory + '/' + filename + '.nii')


