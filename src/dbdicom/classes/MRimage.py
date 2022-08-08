import struct
import numpy as np
from .image import Image

class MRImage(Image):
    """Specific methods for the SOPClass MR Image Storage"""

    def array(self):
        """Read the pixel array from an MR image"""

        ds = self.read().to_pydicom()
        array = ds.pixel_array.astype(np.float32)
        if [0x2005, 0x100E] in ds: # 'Philips Rescale Slope'
            slope = ds[(0x2005, 0x100E)].value
            intercept = ds[(0x2005, 0x100D)].value
            array -= intercept
            array /= slope
        else:
            slope = float(getattr(ds, 'RescaleSlope', 1)) 
            intercept = float(getattr(ds, 'RescaleIntercept', 0)) 
            #array = array * slope + intercept
            array *= slope
            array += intercept
        
        return np.transpose(array)
        
    def set_array(self, array, value_range=None):

        dataset = self.read()
        if dataset is None:
            # TODO: Handle this by creating new dataset from scratch
            raise RuntimeError('Cannot set array: no dataset defined on disk or in memory')
        ds = dataset.to_pydicom() # DataSet needs to inherit this functionality
        if (0x2005, 0x100E) in ds: del ds[0x2005, 0x100E]  # Delete 'Philips Rescale Slope'
        if (0x2005, 0x100D) in ds: del ds[0x2005, 0x100D]
        
        if array.ndim >= 3: # remove spurious dimensions of 1
            array = np.squeeze(array) 
        array = self.clip(array, value_range=value_range)
        array, slope, intercept = self.scale_to_range(array, ds.BitsAllocated)
        array = np.transpose(array)

        maximum = np.amax(array)
        minimum = np.amin(array)
        shape = np.shape(array)

        ds.PixelRepresentation = 0
        ds.SmallestImagePixelValue = int(maximum)
        ds.LargestImagePixelValue = int(minimum)
        ds.RescaleSlope = 1 / slope
        ds.RescaleIntercept = - intercept / slope
#        ds.WindowCenter = (maximum + minimum) / 2
#        ds.WindowWidth = maximum - minimum
        ds.Rows = shape[0]
        ds.Columns = shape[1]
        ds.PixelData = array.tobytes()

        #dataset.set_pydicom(ds)
        self.write(dataset)


    def image_type(self):
        """Determine if an image is Magnitude, Phase, Real or Imaginary image or None"""

        ds = self.read()
        if (0x0043, 0x102f) in ds:
            private_ge = ds[0x0043, 0x102f]
            try: value = struct.unpack('h', private_ge.value)[0]
            except: value = private_ge.value
            if value == 0: return 'MAGNITUDE'
            if value == 1: return 'PHASE'
            if value == 2: return 'REAL'
            if value == 3: return 'IMAGINARY'
        if 'ImageType' in ds:
            type = set(ds.ImageType)
            if set(['M', 'MAGNITUDE']).intersection(type):
                return 'MAGNITUDE'
            if set(['P', 'PHASE']).intersection(type):
                return 'PHASE'
            if set(['R', 'REAL']).intersection(type):
                return 'REAL'
            if set(['I', 'IMAGINARY']).intersection(type):
                return 'IMAGINARY'
        if 'ComplexImageComponent' in ds:
            return ds.ComplexImageComponent


    def signal_type(self):
        """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

        ds = self.read()
        flagWater = False
        flagFat = False
        flagInPhase = False
        flagOutPhase = False
        if hasattr(ds, 'ImageType'):
            type = set(ds.ImageType)
            if set(['W', 'WATER']).intersection(type):
                flagWater = True
            elif set(['F', 'FAT']).intersection(type):# or ('B0' in dataset.ImageType) or ('FIELD_MAP' in dataset.ImageType):
                flagFat = True
            elif set(['IP', 'IN_PHASE']).intersection(type):
                flagInPhase = True
            elif set(['OP', 'OUT_PHASE']).intersection(type):
                flagOutPhase = True
       
        return flagWater, flagFat, flagInPhase, flagOutPhase


