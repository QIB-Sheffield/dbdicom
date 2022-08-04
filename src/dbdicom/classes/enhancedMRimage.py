import numpy as np

from .MRimage import MRImage


class EnhancedMRImage(MRImage):
    """Specific methods for the SOPClass EnhancedMR Image Storage"""

    def array(self):

        if self.__class__.__name__ == 'Record':
            ds = self.read() # only for records, not datasets
        pixelArray = ds.to_pydicom().pixel_array.astype(np.float32)
        frames = ds.PerFrameFunctionalGroupsSequence
        for index, frame in enumerate(frames):
            slice = np.squeeze(pixelArray[index, ...])
            if [0x2005, 0x100E] in ds: # 'Philips Rescale Slope'
                slope = ds[(0x2005, 0x100E)].value
                intercept = ds[(0x2005, 0x100D)].value
                slice = (slice - intercept) / slope
            else:
                transform = frame.PixelValueTransformationSequence[0]
                slope = float(getattr(transform, 'RescaleSlope', 1)) 
                intercept = float(getattr(transform, 'RescaleIntercept', 0)) 
                slice = slice * slope + intercept
            pixelArray[index, ...] = np.transpose(slice)
        
        return pixelArray

    def set_array(self, pixelArray, value_range=None):

        if self.__class__.__name__ == 'Record':
            ds = self.read()
        pixelArray = self.clip(pixelArray, value_range=value_range)
        pixelArray, slope, intercept = self.scale_to_range(pixelArray, ds.BitsAllocated)
        pixelArray = np.transpose(pixelArray, (0, 2, 1))

        maximum = np.amax(pixelArray)
        minimum = np.amin(pixelArray)
        shape = np.shape(pixelArray)

        ds.NumberOfFrames = np.shape(pixelArray)[0]
        del ds.PerFrameFunctionalGroupsSequence[ds.NumberOfFrames:]

        del ds[0x2005, 0x100E]  # Delete 'Philips Rescale Slope'
        del ds[0x2005, 0x100D]
        ds.PixelRepresentation = 0
        ds.SmallestImagePixelValue = int(maximum)
        ds.LargestImagePixelValue = int(minimum)
        ds.RescaleSlope = 1 / slope
        ds.RescaleIntercept = - intercept / slope
        ds.WindowCenter = (maximum + minimum) / 2
        ds.WindowWidth = maximum - minimum
        ds.Rows = shape[0]
        ds.Columns = shape[1]
        ds.PixelData = pixelArray.tobytes()

        if self.__class__.__name__ == 'Record':
            self.write(ds) 

    def signal_type(self):
        """Determine if an image is Water, Fat, In-Phase, Out-phase image or None"""

        if self.__class__.__name__ == 'Record':
            ds = self.read()

        flagWater = False
        flagFat = False
        flagInPhase = False
        flagOutPhase = False
        type = ds.MRImageFrameTypeSequence[0]
        if hasattr(type, 'FrameType'):
            type = set(type.FrameType)
        elif hasattr(type, 'ComplexImageComponent'):
            type = set(type.ComplexImageComponent)
        else:
            return flagWater, flagFat, flagInPhase, flagOutPhase
        if set(['W', 'WATER']).intersection(type):
            flagWater = True
        elif set(['F', 'FAT']).intersection(type):
            flagFat = True
        elif set(['IP', 'IN_PHASE']).intersection(type):
            flagInPhase = True
        elif set(['OP', 'OUT_PHASE']).intersection(type):
            flagOutPhase = True

        return flagWater, flagFat, flagInPhase, flagOutPhase

    def image_type(self):
        """Determine if a dataset is Magnitude, Phase, Real or Imaginary"""

        on_disk = self.on_disk()
        if on_disk: self.read()

        flagMagnitude = []
        flagPhase = []
        flagReal = []
        flagImaginary = []
        for index, singleSlice in enumerate(self.ds.PerFrameFunctionalGroupsSequence):
            sequence = singleSlice.MRImageFrameTypeSequence[0]
            if hasattr(sequence, 'FrameType'):
                type = set(sequence.FrameType)
                if set(['M', 'MAGNITUDE']).intersection(type):
                    flagMagnitude.append(index)
                    continue
                elif set(['P', 'PHASE']).intersection(type):
                    flagPhase.append(index)
                    continue
                elif set(['R', 'REAL']).intersection(type):
                    flagReal.append(index)
                    continue
                elif set(['I', 'IMAGINARY']).intersection(type):
                    flagImaginary.append(index)
                    continue
            if hasattr(sequence, 'ComplexImageComponent'):
                type = set(sequence.ComplexImageComponent)
                if set(['M', 'MAGNITUDE']).intersection(type):
                    flagMagnitude.append(index)
                    continue
                elif set(['P', 'PHASE']).intersection(type):
                    flagPhase.append(index)
                    continue
                elif set(['R', 'REAL']).intersection(type):
                    flagReal.append(index)
                    continue
                elif set(['I', 'IMAGINARY']).intersection(type):
                    flagImaginary.append(index)
                    continue
        if on_disk: self.clear()
        return flagMagnitude, flagPhase, flagReal, flagImaginary

    @property
    def affine_matrix(self):

        on_disk = self.on_disk()
        if on_disk: self.read()
        affineList = list()
        for frame in self.ds.PerFrameFunctionalGroupsSequence:
            affine = affine(
                image_orientation = frame.PlaneOrientationSequence[0].ImageOrientationPatient, 
                image_position = frame.PlanePositionSequence[0].ImagePositionPatient, 
                pixel_spacing = frame.PixelMeasuresSequence[0].PixelSpacing, 
                slice_spacing = frame.PixelMeasuresSequence[0].SpacingBetweenSlices)
            affineList.append(affine)
        if on_disk: self.clear()
        return np.squeeze(np.array(affineList))

    def window(self):
        """Centre and width of the pixel data after applying rescale slope and intercept.
        
        In this case retrieve the centre and width values of the first frame
        """
        on_disk = self.on_disk()
        if on_disk: self.read()
        centre = self.ds.PerFrameFunctionalGroupsSequence[0].FrameVOILUTSequence[0].WindowCenter 
        width = self.ds.PerFrameFunctionalGroupsSequence[0].FrameVOILUTSequence[0].WindowWidth
        if centre is None or width is None:
            array = self.array()
        if centre is None: 
            centre = np.median(array)
        if width is None: 
            p = np.percentile(array, [25, 75])
            width = p[1] - p[0]
        if on_disk: self.clear()
        return centre, width