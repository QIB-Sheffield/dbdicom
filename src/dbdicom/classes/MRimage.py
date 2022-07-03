import struct
import numpy as np
from .image import Image

class MRImage(Image):
    """Specific methods for the SOPClass MR Image Storage"""

    def array(self):
        """Read the pixel array from an MR image"""

        on_disk = self.on_disk()
        if on_disk: self.read()
        if [0x2005, 0x100E] in self.ds: # 'Philips Rescale Slope'
            array = self.ds.pixel_array.astype(np.float32)
            slope = self.ds[(0x2005, 0x100E)].value
            intercept = self.ds[(0x2005, 0x100D)].value
            array -= intercept
            array /= slope
            array = np.transpose(array)
        else:
            array = super().array()
        if on_disk: self.clear()
        return array
        
    def set_array(self, pixelArray, value_range=None):

        on_disk = self.on_disk()
        if on_disk: self.read()
        if self.ds is None:
            # TODO: Handle this by creating new dataset from scratch
            raise RuntimeError('Cannot set array: no dataset defined on disk or in memory')
        if (0x2005, 0x100E) in self.ds: del self.ds[0x2005, 0x100E]  # Delete 'Philips Rescale Slope'
        if (0x2005, 0x100D) in self.ds: del self.ds[0x2005, 0x100D]
        super().set_array(pixelArray, value_range=value_range)
        #self.write()
        if on_disk: 
            self.write()
            self.clear()

    def image_type(self):
        """Determine if an image is Magnitude, Phase, Real or Imaginary image or None"""

        on_disk = self.on_disk()
        if on_disk: self.read()
        if (0x0043, 0x102f) in self.ds:
            private_ge = self.ds[0x0043, 0x102f]
            try: value = struct.unpack('h', private_ge.value)[0]
            except: value = private_ge.value
            if value == 0: return 'MAGNITUDE'
            if value == 1: return 'PHASE'
            if value == 2: return 'REAL'
            if value == 3: return 'IMAGINARY'
        if 'ImageType' in self.ds:
            type = set(self.ds.ImageType)
            if set(['M', 'MAGNITUDE']).intersection(type):
                return 'MAGNITUDE'
            if set(['P', 'PHASE']).intersection(type):
                return 'PHASE'
            if set(['R', 'REAL']).intersection(type):
                return 'REAL'
            if set(['I', 'IMAGINARY']).intersection(type):
                return 'IMAGINARY'
        if 'ComplexImageComponent' in self.ds:
            return self.ds.ComplexImageComponent
        if on_disk: self.clear()

    def signal_type(self):
        """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

        on_disk = self.on_disk()
        if on_disk: self.read()
        flagWater = False
        flagFat = False
        flagInPhase = False
        flagOutPhase = False
        if hasattr(self.ds, 'ImageType'):
            type = set(self.ds.ImageType)
            if set(['W', 'WATER']).intersection(type):
                flagWater = True
            elif set(['F', 'FAT']).intersection(type):# or ('B0' in dataset.ImageType) or ('FIELD_MAP' in dataset.ImageType):
                flagFat = True
            elif set(['IP', 'IN_PHASE']).intersection(type):
                flagInPhase = True
            elif set(['OP', 'OUT_PHASE']).intersection(type):
                flagOutPhase = True
        if on_disk: self.clear()
        return flagWater, flagFat, flagInPhase, flagOutPhase


