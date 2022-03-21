from pydicom.dataset import Dataset
import numpy as np
import datetime


  
class ParametricMap(object):
    def selectParametricMap(self, dicom, imageArray, argument):
        methodName = argument
        method = getattr(self, methodName, lambda: "No valid Parametric Map chosen")
        return method(dicom, imageArray)

    def RGB(self, dicom, imageArray):
        dicom.PhotometricInterpretation = 'RGB'
        dicom.SamplesPerPixel = 3
        dicom.BitsAllocated = 8
        dicom.BitsStored = 8
        dicom.HighBit = 7
        dicom.add_new(0x00280006, 'US', 0) # Planar Configuration
        dicom.RescaleSlope = 1
        dicom.RescaleIntercept = 0
        pixelArray = imageArray.astype(np.uint8) # Should we multiply by 255?
        dicom.WindowCenter = int((np.amax(imageArray) - np.amin(imageArray)) / 2)
        dicom.WindowWidth = np.absolute(int(np.amax(imageArray) - np.amin(imageArray)))
        dicom.PixelData = pixelArray.tobytes()
        return

    def ADC(self, dicom, imageArray):
        # The commented parts are to apply when we decide to include Parametric Map IOD. No readers can deal with this yet
        # dicom.SOPClassUID = '1.2.840.10008.5.1.4.1.1.67'
        dicom.SeriesDescription = "Apparent Diffusion Coefficient (um2/s)"
        dicom.Modality = "RWV"
        dicom.FrameLaterality = "U"
        dicom.DerivedPixelContrast = "ADC"
        dicom.BitsAllocated = 32
        dicom.PixelRepresentation = 1
        dicom.PhotometricInterpretation = "MONOCHROME2"
        dicom.PixelAspectRatio = ["1", "1"] # Need to have a better look at this
        dicom.RescaleSlope = 1
        dicom.RescaleIntercept = 0
        # Rotate the image back to the original orientation
        imageArray = np.transpose(imageArray)
        dicom.Rows = np.shape(imageArray)[-2]
        dicom.Columns = np.shape(imageArray)[-1]
        dicom.WindowCenter = int((np.amax(imageArray) - np.amin(imageArray)) / 2)
        dicom.WindowWidth = np.absolute(int(np.amax(imageArray) - np.amin(imageArray)))
        dicom.FloatPixelData = bytes(imageArray.astype(np.float32).flatten())
        del dicom.PixelData, dicom.BitsStored, dicom.HighBit

        dicom.RealWorldValueMappingSequence = [Dataset(), Dataset(), Dataset(), Dataset()]
        dicom.RealWorldValueMappingSequence[0].QuantityDefinitionSequence = [Dataset(), Dataset()]
        dicom.RealWorldValueMappingSequence[0].QuantityDefinitionSequence[0].ValueType = "CODE"
        dicom.RealWorldValueMappingSequence[0].QuantityDefinitionSequence[1].ConceptCodeSequence = [Dataset(), Dataset(), Dataset()]
        dicom.RealWorldValueMappingSequence[0].QuantityDefinitionSequence[1].ConceptCodeSequence[0].CodeValue = "113041"
        dicom.RealWorldValueMappingSequence[0].QuantityDefinitionSequence[1].ConceptCodeSequence[1].CodingSchemeDesignator = "DCM"
        dicom.RealWorldValueMappingSequence[0].QuantityDefinitionSequence[1].ConceptCodeSequence[2].CodeMeaning = "Apparent Diffusion Coefficient"
        dicom.RealWorldValueMappingSequence[1].MeasurementUnitsCodeSequence = [Dataset(), Dataset(), Dataset()]
        dicom.RealWorldValueMappingSequence[1].MeasurementUnitsCodeSequence[0].CodeValue = "um2/s"
        dicom.RealWorldValueMappingSequence[1].MeasurementUnitsCodeSequence[1].CodingSchemeDesignator = "UCUM"
        dicom.RealWorldValueMappingSequence[1].MeasurementUnitsCodeSequence[2].CodeMeaning = "um2/s"
        dicom.RealWorldValueMappingSequence[2].RealWorldValueSlope = 1
        
        anatomyString = dicom.BodyPartExamined
        saveAnatomicalInfo(anatomyString, dicom.RealWorldValueMappingSequence[3])

        return

    def T2Star(self, dicom, imageArray):
        dicom.PixelSpacing = [3, 3] # find a mechanism to pass reconstruct pixel here
        return

    def SEG(self, dicom, imageArray):
        #dicom.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4' # WILL NOT BE USED HERE - This is for PACS. There will be another one for DICOM Standard
        # The commented parts are to apply when we decide to include SEG IOD. No readers can deal with this yet
        dicom.BitsAllocated = 8 # According to Federov DICOM Standard this should be 1-bit
        dicom.BitsStored = 8
        dicom.HighBit = 7
        #dicom.SmallestImagePixelValue = 0
        #dicom.LargestImagePixelValue = int(np.amax(imageArray)) # max 255
        dicom.add_new('0x00280106', 'US', 0) # Minimum
        dicom.add_new('0x00280107', 'US', int(np.amax(imageArray))) # Maximum
        dicom.PixelRepresentation = 0
        dicom.SamplesPerPixel = 1
        #dicom.WindowCenter = 0.5
        #dicom.WindowWidth = 1.1
        dicom.add_new('0x00281050', 'DS', 0.5) # WindowCenter
        dicom.add_new('0x00281051', 'DS', 1.1) # WindowWidth
        #dicom.RescaleIntercept = 0
        #dicom.RescaleSlope = 1
        dicom.add_new('0x00281052', 'DS', 0) # RescaleIntercept
        dicom.add_new('0x00281053', 'DS', 1) # RescaleSlope
        dicom.LossyImageCompression = '00'
        pixelArray = np.transpose(imageArray.astype(np.uint8)) # Should we multiply by 255?
        dicom.PixelData = pixelArray.tobytes()

        dicom.Modality = 'SEG'
        dicom.SegmentationType = 'FRACTIONAL'
        dicom.MaximumFractionalValue = int(np.amax(imageArray)) # max 255
        dicom.SegmentationFractionalType = 'OCCUPANCY'

        # Segment Labels
        if hasattr(dicom, "ImageComments"):
            dicom.ContentDescription = dicom.ImageComments.split('_')[-1] # 'Image segmentation'
            segment_numbers = np.unique(pixelArray)
            segment_dictionary = dict(list(enumerate(segment_numbers)))
            segment_label = dicom.ImageComments.split('_')[-1]
            segment_dictionary[0] = 'Background'
            segment_dictionary[1] = segment_label
            for key in segment_dictionary:
                dicom.SegmentSequence = [Dataset(), Dataset(), Dataset(), Dataset(), Dataset(), Dataset()]
                dicom.SegmentSequence[0].SegmentAlgorithmType = 'MANUAL'
                dicom.SegmentSequence[1].SegmentNumber = key
                dicom.SegmentSequence[2].SegmentDescription = str(segment_dictionary[key])
                dicom.SegmentSequence[3].SegmentLabel = str(segment_dictionary[key])
                dicom.SegmentSequence[4].SegmentAlgorithmName = "Weasel"
                if hasattr(dicom, "BodyPartExamined"):
                    anatomyString = dicom.BodyPartExamined
                    saveAnatomicalInfo(anatomyString, dicom.SegmentSequence[5])
        else:
            dicom.ContentDescription = "Mask with no label"

        return

    def Registration(self, dicom, imageArray):
        dicom.Modality = "REG"
        return

    def Signal(self, dicom, imageArray):
        dicom.Modality = "RWV"
        dicom.DerivedPixelContrast = "GraphPlot"
        dicom.PhotometricInterpretation = "MONOCHROME2"
        dicom.RescaleSlope = 1
        dicom.RescaleIntercept = 0
        imageArray = np.transpose(imageArray.astype(np.float32))
        center = (np.amax(imageArray) + np.amin(imageArray)) / 2
        width = np.amax(imageArray) - np.amin(imageArray)
        dicom.add_new('0x00281050', 'DS', center)
        dicom.add_new('0x00281051', 'DS', width)
        dicom.BitsAllocated = 32
        dicom.Rows = np.shape(imageArray)[0]
        dicom.Columns = np.shape(imageArray)[1]
        dicom.FloatPixelData = bytes(imageArray.flatten())
        del dicom.PixelData, dicom.BitsStored, dicom.HighBit
        return

# Could insert a method regarding ROI colours, like in ITK-SNAP???
def saveAnatomicalInfo(anatomyString, dicom):
    try:
        # FOR NOW, THE PRIORITY WILL BE ON KIDNEY
        if "KIDNEY" or "ABDOMEN" in anatomyString.upper():
            dicom.AnatomicRegionSequence = [Dataset(), Dataset(), Dataset()]
            dicom.AnatomicRegionSequence[0].CodeValue = "T-71000"
            dicom.AnatomicRegionSequence[1].CodingSchemeDesignator = "SRT"
            dicom.AnatomicRegionSequence[2].CodeMeaning = "Kidney"
        elif "LIVER" in anatomyString.upper():
            dicom.AnatomicRegionSequence = [Dataset(), Dataset(), Dataset()]
            dicom.AnatomicRegionSequence[0].CodeValue = "T-62000"
            dicom.AnatomicRegionSequence[1].CodingSchemeDesignator = "SRT"
            dicom.AnatomicRegionSequence[2].CodeMeaning = "Liver"
        elif "PROSTATE" in anatomyString.upper():
            dicom.AnatomicRegionSequence = [Dataset(), Dataset(), Dataset()]
            dicom.AnatomicRegionSequence[0].CodeValue = "T-9200B"
            dicom.AnatomicRegionSequence[1].CodingSchemeDesignator = "SRT"
            dicom.AnatomicRegionSequence[2].CodeMeaning = "Prostate"      
        elif "BODY" in anatomyString.upper():
            dicom.AnatomicRegionSequence = [Dataset(), Dataset(), Dataset()]
            dicom.AnatomicRegionSequence[0].CodeValue = "P5-0905E"
            dicom.AnatomicRegionSequence[1].CodingSchemeDesignator = "LN"
            dicom.AnatomicRegionSequence[2].CodeMeaning = "MRI whole body"
    except:
        pass
    return

def editDicom(newDicom, imageArray, parametricMap):

    callCase = ParametricMap()
    callCase.selectParametricMap(newDicom, imageArray, parametricMap)

    dt = datetime.datetime.now()
    timeStr = dt.strftime('%H%M%S')  # long format with micro seconds
    newDicom.PerformedProcedureStepStartDate = dt.strftime('%Y%m%d')
    newDicom.PerformedProcedureStepStartTime = timeStr
    newDicom.PerformedProcedureStepDescription = "Post-processing application"

    return newDicom

    # Series, Instance and Class for Reference
    #newDicom.ReferencedSeriesSequence = [Dataset(), Dataset()]
    #newDicom.ReferencedSeriesSequence[0].SeriesInstanceUID = dicom_data.SeriesInstanceUID
    #newDicom.ReferencedSeriesSequence[1].ReferencedInstanceSequence = [Dataset(), Dataset()]
    #newDicom.ReferencedSeriesSequence[1].ReferencedInstanceSequence[0].ReferencedSOPClassUID = dicom_data.SOPClassUID
    #newDicom.ReferencedSeriesSequence[1].ReferencedInstanceSequence[1].ReferencedSOPInstanceUID = dicom_data.SOPInstanceUID

# rwv_sequence = Sequence()
        # dicom.RealWorldValueMappingSequence = rwv_sequence
        # rwv_slope = Dataset()
        # rwv_slope.RealWorldValueSlope = 1
        # rwv_sequence.append(rwv_slope)

        # quantity_def = Dataset()
        # quantity_def_sequence = Sequence()
        # quantity_def.QuantityDefinitionSequence = quantity_def_sequence
        # value_type = Dataset()
        # value_type.ValueType = "CODE"
        # quantity_def_sequence.append(value_type)
        # concept_code = Dataset()
        # concept_code_sequence = Sequence()
        # concept_code.ConceptCodeSequence = concept_code_sequence
        # code_code = Dataset()
        # code_code.CodeValue = "113041"
        # code_code.CodingSchemeDesignator = "DCM"
        # code_code.CodeMeaning = "Apparent Diffusion Coefficient"
        # concept_code_sequence.append(code_code)
        # rwv_sequence.append(quantity_def)

        # measure_units = Dataset()
        # measure_units_sequence = Sequence()
        # measure_units.MeasurementUnitsCodeSequence = measure_units_sequence
        # measure_code = Dataset()
        # measure_code.CodeValue = "um2/s"
        # measure_code.CodingSchemeDesignator = "UCUM"
        # measure_code.CodeMeaning = "um2/s"
        # measure_units_sequence.append(measure_code)
        # rwv_sequence.append(measure_units)