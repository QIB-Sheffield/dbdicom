__all__ = ['QImage']

import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib import cm
from nibabel.affines import apply_affine
from PyQt5 import QtGui

from .instance import Instance

class Image(Instance):
    """Specific methods for the SOPClass MR Image Storage"""

    def _initialize(self, ref_ds=None):
        """Initialize the attributes relevant for the Images"""

        super()._initialize(ref_ds)
        
        self.ImageType.insert(0, "DERIVED")

    def array(self):
        """Read the pixel array from an image"""

        on_disk = self.on_disk()
        if on_disk: self.read()
        if self.ds is None: return
        array = self.ds.pixel_array.astype(np.float32)
        slope = float(getattr(self.ds, 'RescaleSlope', 1)) 
        intercept = float(getattr(self.ds, 'RescaleIntercept', 0)) 
        #array = array * slope + intercept
        array *= slope
        array += intercept
        array = np.transpose(array)
        if on_disk: self.clear()
        return array

    def set_array(self, array, value_range=None):

        on_disk = self.on_disk()
        if on_disk: self.read()
        
        if array.ndim >= 3: # remove spurious dimensions of 1
            array = np.squeeze(array) 
        array = self.clip(array, value_range=value_range)
        array, slope, intercept = self.scale_to_range(array, self.ds.BitsAllocated)
        array = np.transpose(array)

        maximum = np.amax(array)
        minimum = np.amin(array)
        shape = np.shape(array)

        self.ds.PixelRepresentation = 0
        self.ds.SmallestImagePixelValue = int(maximum)
        self.ds.LargestImagePixelValue = int(minimum)
        self.ds.RescaleSlope = 1 / slope
        self.ds.RescaleIntercept = - intercept / slope
#        self.ds.WindowCenter = (maximum + minimum) / 2
#        self.ds.WindowWidth = maximum - minimum
        self.ds.Rows = shape[0]
        self.ds.Columns = shape[1]
        self.ds.PixelData = array.tobytes()
        if on_disk: 
            self.write()
            self.clear()

#    def write_array(self, pixelArray, value_range=None): # obsolete - remove
#        """Write the pixel array to disk"""
#        self.set_array(pixelArray, value_range=value_range)
#        self.write()

    def clip(self, array, value_range = None):

        array[np.isnan(array)] = 0
        if value_range is None:
            finite = array[np.isfinite(array)]
            value_range = [np.amin(finite), np.amax(finite)]
        return np.clip(array, value_range[0], value_range[1])

    def scale_to_range(self, array, bits_allocated):
            
    #    target = np.power(2, bits_allocated) - 1
        target = 2.0**bits_allocated - 1
        maximum = np.amax(array)
        minimum = np.amin(array)
        if maximum == minimum:
            slope = 1
        else:
            slope = target / (maximum - minimum)
        intercept = -slope * minimum
        # array = slope * (array - minimum)
        array = slope * array + intercept

        if bits_allocated == 8:
            return array.astype(np.uint8), slope, intercept
        if bits_allocated == 16:
            return array.astype(np.uint16), slope, intercept
        if bits_allocated == 32:
            return array.astype(np.uint32), slope, intercept
        if bits_allocated == 64:
            return array.astype(np.uint64), slope, intercept

    def zeros(self):

        array = np.zeros((self.Rows, self.Columns))
        new = self.copy()
    #    new.write_array(array)
        new.set_array(array)
        new.write()
        return new

    def map_onto(self, target):
        """Map non-zero image pixels onto a target image.
        
        Overwrite pixel values in the target"""

        # Create a coordinate array of non-zero pixels
        coords = np.transpose(np.where(self.array() != 0)) 
        coords = [[coord[0], coord[1], 0] for coord in coords] 
        coords = np.array(coords)

        # Determine coordinate transformation matrix
        affineSource = self.affine_matrix()
        affineTarget = target.affine_matrix()
        sourceToTarget = np.linalg.inv(affineTarget).dot(affineSource)

        # Apply coordinate transformation
        coords = apply_affine(sourceToTarget, coords)
        coords = np.round(coords, 3).astype(int)
        x = tuple([coord[0] for coord in coords if coord[2] == 0])
        y = tuple([coord[1] for coord in coords if coord[2] == 0])

        # Set values in the target image
        # Note - replace by actual values rather than 1 & 0.
        result = target.zeros()
        in_memory = self.in_memory()
        if in_memory: result.read()
        pixelArray = result.array()
        pixelArray[(x, y)] = 1.0
        result.set_array(pixelArray)
        if not in_memory: 
            result.write()
            result.clear()

        return result

    def affine_matrix(self):
        """Affine transformation matrix for a DICOM image"""

        on_disk = self.on_disk()
        if on_disk: self.read()

        image_orientation = self.ds.ImageOrientationPatient
        image_position = self.ds.ImagePositionPatient
        pixel_spacing = self.ds.PixelSpacing
        slice_spacing = self.ds.SliceThickness            

        row_spacing = pixel_spacing[0]
        column_spacing = pixel_spacing[1]

        row_cosine = np.array(image_orientation[:3])
        column_cosine = np.array(image_orientation[3:])
        slice_cosine = np.cross(row_cosine, column_cosine)

        affine = np.identity(4, dtype=np.float32)
        affine[:3, 0] = row_cosine * column_spacing
        affine[:3, 1] = column_cosine * row_spacing
        affine[:3, 2] = slice_cosine * slice_spacing
        affine[:3, 3] = image_position

        if on_disk: self.clear()

        return affine

    def get_colormap(self):
        """Returns the colormap if there is any."""

        on_disk = self.on_disk()
        if on_disk: self.read()
        ds = self.ds

        lut = None
        if hasattr(ds, 'ContentLabel'):
            if ds.PhotometricInterpretation == 'PALETTE COLOR':
                colormap = ds.ContentLabel
            elif 'MONOCHROME' in ds.PhotometricInterpretation:
                colormap = 'gray'
        elif len(ds.dir("PaletteColor"))>=3 and ds.PhotometricInterpretation == 'PALETTE COLOR':
            colormap = 'custom'
            lut = self.get_lut()
        else:
            colormap = 'gray' # default

        if on_disk: self.clear()
        return colormap, lut  

    def get_lut(self):
        
        on_disk = self.on_disk()
        if on_disk: self.read()
        ds = self.ds

        redColour = list(ds.RedPaletteColorLookupTableData)
        greenColour = list(ds.GreenPaletteColorLookupTableData)
        blueColour = list(ds.BluePaletteColorLookupTableData)
        redLut = list(struct.unpack('<' + ('H' * ds.RedPaletteColorLookupTableDescriptor[0]), bytearray(redColour)))
        greenLut = list(struct.unpack('<' + ('H' * ds.GreenPaletteColorLookupTableDescriptor[0]), bytearray(greenColour)))
        blueLut = list(struct.unpack('<' + ('H' * ds.BluePaletteColorLookupTableDescriptor[0]), bytearray(blueColour)))
        colours = np.transpose([redLut, greenLut, blueLut])
        normaliseFactor = int(np.power(2, ds.RedPaletteColorLookupTableDescriptor[2]))
        # Fast ColourTable loading
        colourTable = np.around(colours/normaliseFactor, decimals = 2)
        indexes = np.unique(colourTable, axis=0, return_index=True)[1]
        lut = [colourTable[index].tolist() for index in sorted(indexes)]
        # Full / Complete Colourmap - takes 20 seconds to load each image
        # lut = (colours/normaliseFactor).tolist()   
        if on_disk: self.clear()
        return lut      

    def set_colormap(self, colormap=None, levels=None):
        """Set the colour table of the image."""

        on_disk = self.on_disk()
        if on_disk: self.read()
        ds = self.ds

        #and (colormap != 'gray') removed from If statement below, so as to save gray colour tables
        if (colormap == 'gray'):
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.ContentLabel = ''
            if hasattr(ds, 'RedPaletteColorLookupTableData'):
                del (ds.RGBLUTTransferFunction, ds.RedPaletteColorLookupTableData,
                    ds.GreenPaletteColorLookupTableData, ds.BluePaletteColorLookupTableData,
                    ds.RedPaletteColorLookupTableDescriptor, ds.GreenPaletteColorLookupTableDescriptor,
                    ds.BluePaletteColorLookupTableDescriptor)
        if ((colormap is not None)  and (colormap != 'custom') and (colormap != 'gray') 
            and (colormap != 'default') and isinstance(colormap, str)):
            ds.PhotometricInterpretation = 'PALETTE COLOR'
            ds.RGBLUTTransferFunction = 'TABLE'
            ds.ContentLabel = colormap
            stringType = 'US' # ('SS' if minValue < 0 else 'US')
            ds.PixelRepresentation = 0 # (1 if minValue < 0 else 0)
            pixelArray = ds.pixel_array
            minValue = int(np.amin(pixelArray))
            maxValue = int(np.amax(pixelArray))
            numberOfValues = int(maxValue - minValue)
            arrayForRGB = np.arange(0, numberOfValues)
            colorsList = cm.ScalarMappable(cmap=colormap).to_rgba(np.array(arrayForRGB), bytes=False)
            totalBytes = ds.BitsAllocated
            ds.add_new('0x00281101', stringType, [numberOfValues, minValue, totalBytes])
            ds.add_new('0x00281102', stringType, [numberOfValues, minValue, totalBytes])
            ds.add_new('0x00281103', stringType, [numberOfValues, minValue, totalBytes])
            ds.RedPaletteColorLookupTableData = bytes(np.array([int((np.power(
                2, totalBytes) - 1) * value) for value in colorsList[:, 0].flatten()]).astype('uint'+str(totalBytes)))
            ds.GreenPaletteColorLookupTableData = bytes(np.array([int((np.power(
                2, totalBytes) - 1) * value) for value in colorsList[:, 1].flatten()]).astype('uint'+str(totalBytes)))
            ds.BluePaletteColorLookupTableData = bytes(np.array([int((np.power(
                2, totalBytes) - 1) * value) for value in colorsList[:, 2].flatten()]).astype('uint'+str(totalBytes)))
        if levels is not None:
            ds.WindowCenter = levels[0]
            ds.WindowWidth = levels[1]
        if on_disk: self.clear()

    def export_as_nifti(self, directory=None, filename=None):
        """Export 2D pixel Array in nifty format"""

        on_disk = self.on_disk()
        if on_disk: self.read()
        ds = self.ds
        if directory is None: 
            directory = self.directory(message='Please select a folder for the nifty data')
        if filename is None:
            filename = self.SeriesDescription
        dicomHeader = nib.nifti1.Nifti1DicomExtension(2, ds)
        niftiObj = nib.Nifti1Instance(np.flipud(np.rot90(np.transpose(self.array()))), affine=self.affine)
        # The transpose is necessary in this case to be in line with the rest of WEASEL.
        niftiObj.header.extensions.append(dicomHeader)
        nib.save(niftiObj, directory + '/' + filename + '.nii.gz')
        if on_disk: self.clear()

    def export_as_csv(self, directory=None, filename=None, columnHeaders=None):
        """Export 2D pixel Array in csv format"""

        if directory is None: 
            directory = self.directory(message='Please select a folder for the csv data')
        if filename is None:
            filename = self.SeriesDescription
        filename = os.path.join(directory, filename + '.csv')
        table = self.array()
        if columnHeaders is None:
            columnHeaders = []
            counter = 0
            for _ in table:
                counter += 1
                columnHeaders.append("Column" + str(counter))
        df = pd.DataFrame(np.transpose(table), columns=columnHeaders)
        df.to_csv(filename, index=False) 

    def export_as_png(self, fileName):
        """Export image in png format."""

        colourTable, _ = self.get_colormap()
        pixelArray = np.transpose(self.array())
        centre, width = self.window()
        minValue = centre - width/2
        maxValue = centre + width/2
        cmap = plt.get_cmap(colourTable)
        plt.imshow(pixelArray, cmap=cmap)
        plt.clim(int(minValue), int(maxValue))
        cBar = plt.colorbar()
        cBar.minorticks_on()
        plt.savefig(fname=fileName + '_' + self.label() + '.png')
        plt.close()

    def window(self):
        """Centre and width of the pixel data after applying rescale slope and intercept"""

        on_disk = self.on_disk()
        if on_disk: self.read()
        if 'WindowCenter' in self.ds: centre = self.ds.WindowCenter
        if 'WindowWidth' in self.ds: width = self.ds.WindowWidth
        if centre is None or width is None:
            array = self.array()
        if centre is None: 
            centre = np.median(array)
        if width is None: 
            p = np.percentile(array, [25, 75])
            width = p[1] - p[0]
        if on_disk: self.clear()
        return centre, width

    def QImage(self):

        array = self.array()
        return QImage(array, width=self.WindowWidth, center=self.WindowCenter)


def QImage(array, width=None, center=None):

    if (width is None) or (center is None):
        max = np.amax(array)
        min = np.amin(array)
    if width is None: width = max-min
    if center is None: center = (max-min)/2

    imgData, alpha = _makeARGB(
        data = array, 
        levels = [center-width/2, center+width/2],
    )
    return _makeQImage(imgData, alpha)


# HELPER FUNCTIONS ADAPTED FROM pyQtGraph


def _makeARGB(data, lut=None, levels=None, scale=None, useRGBA=False): 
    """ 
    Convert an array of values into an ARGB array suitable for building QImages
    
    Returns the ARGB array (unsigned byte) and a boolean indicating whether
    there is alpha channel data. This is a two stage process:
    
        1) Rescale the data based on the values in the *levels* argument (min, max).
        2) Determine the final output by passing the rescaled values through a
           lookup table.
   
    Both stages are optional.
    
    ============== ==================================================================================
    **Arguments:**
    data           numpy array of int/float types. If 
    levels         List [min, max]; optionally rescale data before converting through the
                   lookup table. The data is rescaled such that min->0 and max->*scale*::
                   
                      rescaled = (clip(data, min, max) - min) * (*scale* / (max - min))
                   
                   It is also possible to use a 2D (N,2) array of values for levels. In this case,
                   it is assumed that each pair of min,max values in the levels array should be 
                   applied to a different subset of the input data (for example, the input data may 
                   already have RGB values and the levels are used to independently scale each 
                   channel). The use of this feature requires that levels.shape[0] == data.shape[-1].
    scale          The maximum value to which data will be rescaled before being passed through the 
                   lookup table (or returned if there is no lookup table). By default this will
                   be set to the length of the lookup table, or 255 if no lookup table is provided.
    lut            Optional lookup table (array with dtype=ubyte).
                   Values in data will be converted to color by indexing directly from lut.
                   The output data shape will be input.shape + lut.shape[1:].
                   Lookup tables can be built using ColorMap or GradientWidget.
    useRGBA        If True, the data is returned in RGBA order (useful for building OpenGL textures). 
                   The default is False, which returns in ARGB order for use with QImage 
                   (Note that 'ARGB' is a term used by the Qt documentation; the *actual* order 
                   is BGRA).
    ============== ==================================================================================
    """

    if data.ndim not in (2, 3):
        raise TypeError("data must be 2D or 3D")
    if data.ndim == 3 and data.shape[2] > 4:
        raise TypeError("data.shape[2] must be <= 4")
    
    if lut is not None and not isinstance(lut, np.ndarray):
        lut = np.array(lut)
    
    if levels is None:
        # automatically decide levels based on data dtype
        if data.dtype.kind == 'u':
            levels = np.array([0, 2**(data.itemsize*8)-1])
        elif data.dtype.kind == 'i':
            s = 2**(data.itemsize*8 - 1)
            levels = np.array([-s, s-1])
        elif data.dtype.kind == 'b':
            levels = np.array([0,1])
        else:
            raise Exception('levels argument is required for float input types')
    if not isinstance(levels, np.ndarray):
        levels = np.array(levels)
    if levels.ndim == 1:
        if levels.shape[0] != 2:
            raise Exception('levels argument must have length 2')
    elif levels.ndim == 2:
        if lut is not None and lut.ndim > 1:
            raise Exception('Cannot make ARGB data when both levels and lut have ndim > 2')
        if levels.shape != (data.shape[-1], 2):
            raise Exception('levels must have shape (data.shape[-1], 2)')
    else:
        raise Exception("levels argument must be 1D or 2D (got shape=%s)." % repr(levels.shape))

    # Decide on maximum scaled value
    if scale is None:
        if lut is not None:
            scale = lut.shape[0] - 1
        else:
            scale = 255.

    # Decide on the dtype we want after scaling
    if lut is None:
        dtype = np.ubyte
    else:
        dtype = np.min_scalar_type(lut.shape[0]-1)
            
    # Apply levels if given
    if levels is not None:
        if isinstance(levels, np.ndarray) and levels.ndim == 2:
            # we are going to rescale each channel independently
            if levels.shape[0] != data.shape[-1]:
                raise Exception("When rescaling multi-channel data, there must be the same number of levels as channels (data.shape[-1] == levels.shape[0])")
            newData = np.empty(data.shape, dtype=int)
            for i in range(data.shape[-1]):
                minVal, maxVal = levels[i]
                if minVal == maxVal:
                    maxVal += 1e-16
                newData[...,i] = _rescaleData(data[...,i], scale/(maxVal-minVal), minVal, dtype=dtype)
            data = newData
        else:
            # Apply level scaling unless it would have no effect on the data
            minVal, maxVal = levels
            if minVal != 0 or maxVal != scale:
                if minVal == maxVal:
                    maxVal += 1e-16
                data = _rescaleData(data, scale/(maxVal-minVal), minVal, dtype=dtype)
            
    # apply LUT if given
    if lut is not None:
        data = _applyLookupTable(data, lut)
    else:
        if data.dtype is not np.ubyte:
            data = np.clip(data, 0, 255).astype(np.ubyte)

    # this will be the final image array
    imgData = np.empty(data.shape[:2]+(4,), dtype=np.ubyte)

    # decide channel order
    if useRGBA:
        order = [0,1,2,3] # array comes out RGBA
    else:
        order = [2,1,0,3] # for some reason, the colors line up as BGR in the final image.
        
    # copy data into image array
    if data.ndim == 2:
        # This is tempting:
        #   imgData[..., :3] = data[..., np.newaxis]
        # ..but it turns out this is faster:
        for i in range(3):
            imgData[..., i] = data
    elif data.shape[2] == 1:
        for i in range(3):
            imgData[..., i] = data[..., 0]
    else:
        for i in range(0, data.shape[2]):
            imgData[..., i] = data[..., order[i]] 
    
    # add opaque alpha channel if needed
    if data.ndim == 2 or data.shape[2] == 3:
        alpha = False
        imgData[..., 3] = 255
    else:
        alpha = True

    return imgData, alpha


def _makeQImage(imgData, alpha=None, copy=True, transpose=True):
    """
    Turn an ARGB array into QImage. 'd
    By default, the data is copied; changes to the array will not
    be reflected in the image. The image will be given aata' attribute
    pointing to the array which shares its data to prevent python
    freeing that memory while the image is in use.
    
    ============== ===================================================================
    **Arguments:**
    imgData        Array of data to convert. Must have shape (width, height, 3 or 4) 
                   and dtype=ubyte. The order of values in the 3rd axis must be 
                   (b, g, r, a).
    alpha          If True, the QImage returned will have format ARGB32. If False,
                   the format will be RGB32. By default, _alpha_ is True if
                   array.shape[2] == 4.
    copy           If True, the data is copied before converting to QImage.
                   If False, the new QImage points directly to the data in the array.
                   Note that the array must be contiguous for this to work
                   (see numpy.ascontiguousarray).
    transpose      If True (the default), the array x/y axes are transposed before 
                   creating the image. Note that Qt expects the axes to be in 
                   (height, width) order whereas pyqtgraph usually prefers the 
                   opposite.
    ============== ===================================================================    
    """
    ## create QImage from buffer
    
    ## If we didn't explicitly specify alpha, check the array shape.
    if alpha is None:
        alpha = (imgData.shape[2] == 4)
        
    copied = False
    if imgData.shape[2] == 3:  ## need to make alpha channel (even if alpha==False; QImage requires 32 bpp)
        if copy is True:
            d2 = np.empty(imgData.shape[:2] + (4,), dtype=imgData.dtype)
            d2[:,:,:3] = imgData
            d2[:,:,3] = 255
            imgData = d2
            copied = True
        else:
            raise Exception('Array has only 3 channels; cannot make QImage without copying.')
    
    if alpha:
        imgFormat = QtGui.QImage.Format_ARGB32
    else:
        imgFormat = QtGui.QImage.Format_RGB32
        
    if transpose:
        imgData = imgData.transpose((1, 0, 2))  ## QImage expects the row/column order to be opposite

    if not imgData.flags['C_CONTIGUOUS']:
        if copy is False:
            extra = ' (try setting transpose=False)' if transpose else ''
            raise Exception('Array is not contiguous; cannot make QImage without copying.'+extra)
        imgData = np.ascontiguousarray(imgData)
        copied = True
        
    if copy is True and copied is False:
        imgData = imgData.copy()       
    try:
        img = QtGui.QImage(imgData.ctypes.data, imgData.shape[1], imgData.shape[0], imgFormat)
    except:
        img = QtGui.QImage(memoryview(imgData), imgData.shape[1], imgData.shape[0], imgFormat)
                
    img.data = imgData
    
    return img
    
def _applyLookupTable(data, lut):
    """
    Uses values in *data* as indexes to select values from *lut*.
    The returned data has shape data.shape + lut.shape[1:]
    """
    if data.dtype.kind not in ('i', 'u'):
        data = data.astype(int)
    
    return np.take(lut, data, axis=0, mode='clip')  


def _rescaleData(data, scale, offset, dtype=None, clip=None):
    """Return data rescaled and optionally cast to a new dtype::
    
        data => (data-offset) * scale
        
    """
    if dtype is None:
        dtype = data.dtype
    else:
        dtype = np.dtype(dtype)
    
    try:
        newData = np.empty((data.size,), dtype=dtype)
        flat = np.ascontiguousarray(data).reshape(data.size)
        newData = (flat - offset)*scale 
        if dtype != dtype: 
            newData = newData.astype(dtype)
        data = newData.reshape(data.shape)
    except:
        
        d2 = data - float(offset)
        d2 *= scale
        
        # Clip before converting dtype to avoid overflow
        if dtype.kind in 'ui':
            lim = np.iinfo(dtype)
            if clip is None:
                # don't let rescale cause integer overflow
                d2 = np.clip(d2, lim.min, lim.max)
            else:
                d2 = np.clip(d2, max(clip[0], lim.min), min(clip[1], lim.max))
        else:
            if clip is not None:
                d2 = np.clip(d2, *clip)
        data = d2.astype(dtype)
    return data
