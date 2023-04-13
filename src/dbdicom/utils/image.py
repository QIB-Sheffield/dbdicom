import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import affine_transform


def multislice_affine_transform(array_source, affine_source, output_affine, slice_thickness=None, **kwargs):
    """Generalization of scipy's affine transform.
    
    This version also works when the source array is 2D and when it is multislice 2D (ie. slice thickness < slice spacing).
    In these scenarios each slice is first reshaped into a volume with provided slice thickness and mapped separately.
    """
    
    slice_spacing = np.linalg.norm(affine_source[:3,2])

    # Single-slice 2D sequence
    if array_source.shape[2] == 1:
        return _map_multislice_array(array_source, affine_source, output_affine, **kwargs)

    # Multi-slice 2D sequence
    elif slice_spacing != slice_thickness:
        return _map_multislice_array(array_source, affine_source, output_affine, slice_thickness=slice_thickness, **kwargs)

    # 3D volume sequence
    else:
        return _map_volume_array(array_source, affine_source, output_affine, **kwargs)



def _map_multislice_array(array, affine, output_affine, output_shape=None, slice_thickness=None, mask=False, label=False, cval=0):

    # Turn each slice into a volume and map as volume.
    array_mapped = None
    for z in range(array.shape[2]):
        array_z, affine_z = _volume_from_slice(array, affine, z, slice_thickness=slice_thickness)
        array_mapped_z = _map_volume_array(array_z, affine_z, output_affine, output_shape=output_shape, cval=cval)
        if array_mapped is None:
            array_mapped = array_mapped_z
        else:
            array_mapped += array_mapped_z

    # If source is a mask array, set values to [0,1].
    if mask:
        array_mapped[array_mapped > 0.5] = 1
        array_mapped[array_mapped <= 0.5] = 0
    elif label:
        array_mapped = np.around(array_mapped)

    return array_mapped


def _volume_from_slice(array, affine, z, slice_thickness=None):

    # Reshape array to 4D (x,y,z + remainder)
    shape = array.shape
    nk = np.prod(shape[3:])
    array = array.reshape(shape[:3] + (nk,))
    
    # Extract a 2D array
    array_z = array[:,:,z,:]
    array_z = array_z[:,:,np.newaxis,:]

    # Duplicate the array in the z-direction to create 2 slices.
    nz = 2
    array_z = np.repeat(array_z, nz, axis=2)

    # Reshape to original nr of dimensions
    array_z = array_z.reshape(shape[:2] + (nz,) + shape[3:])

    # Offset the slice position accordingly
    affine_z = affine.copy()
    affine_z[:3,3] += z*affine_z[:3,2]

    # Set the slice spacing to equal the slice thickness
    if slice_thickness is not None:
        slice_spacing = np.linalg.norm(affine_z[:3,2])
        affine_z[:3,2] *= slice_thickness/slice_spacing

    # Offset the slice position by half of the slice thickness.
    affine_z[:3,3] -= affine_z[:3,2]/2

    return array_z, affine_z


def _map_volume_array(array, affine, output_affine, output_shape=None, mask=False, label=False, cval=0):

    shape = array.shape
    if shape[2] == 1:
        msg = 'This function only works for an array with at least 2 slices'
        raise ValueError(msg)

    # Get transformation matrix
    source_to_target = np.linalg.inv(affine).dot(output_affine)
    source_to_target = np.around(source_to_target, 3) # remove round-off errors in the inversion

    # Reshape array to 4D (x,y,z + remainder)
    if output_shape is None:
        output_shape = shape[:3]
    nk = np.prod(shape[3:])
    output = np.empty(output_shape + (nk,))
    array = array.reshape(shape[:3] + (nk,))

    #Perform transformation
    for k in range(nk):
        output[:,:,:,k] = affine_transform(
            array[:,:,:,k],
            matrix = source_to_target[:3,:3],
            offset = source_to_target[:3,3],
            output_shape = output_shape,
            cval = cval,
            order = 0 if mask else 3,
        )

    # If source is a mask array, set values to [0,1]
    if mask:
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

    # If source is a label array, round to integers
    elif label:
        output = np.around(output)

    return output.reshape(output_shape + shape[3:])



# https://discovery.ucl.ac.uk/id/eprint/10146893/1/geometry_medim.pdf

def interpolate3d_scale(array, scale=2):

    array, _ = interpolate3d_isotropic(array, [1,1,1], isotropic_spacing=1/scale)
    return array


def interpolate3d_isotropic(array, spacing, isotropic_spacing=None):

    if isotropic_spacing is None:
        isotropic_spacing = np.amin(spacing)

    # Get x, y, z coordinates for array
    nx = array.shape[0]
    ny = array.shape[1]
    nz = array.shape[2]
    Lx = (nx-1)*spacing[0]
    Ly = (ny-1)*spacing[1]
    Lz = (nz-1)*spacing[2]
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    # Get x, y, z coordinates for isotropic array
    nxi = 1 + np.floor(Lx/isotropic_spacing)
    nyi = 1 + np.floor(Ly/isotropic_spacing)
    nzi = 1 + np.floor(Lz/isotropic_spacing)
    Lxi = (nxi-1)*isotropic_spacing
    Lyi = (nyi-1)*isotropic_spacing
    Lzi = (nzi-1)*isotropic_spacing
    xi = np.linspace(0, Lxi, nxi.astype(int))
    yi = np.linspace(0, Lyi, nyi.astype(int))
    zi = np.linspace(0, Lzi, nzi.astype(int))

    # Interpolate to isotropic
    ri = np.meshgrid(xi,yi,zi, indexing='ij')
    array = interpn((x,y,z), array, np.stack(ri, axis=-1))
    return array, isotropic_spacing


def bounding_box(
    image_orientation,  # ImageOrientationPatient (assume same for all slices)
    image_positions,    # ImagePositionPatient for all slices
    pixel_spacing,      # PixelSpacing (assume same for all slices)
    rows,               # Number of rows
    columns):           # Number of columns   

    """
    Calculate the bounding box of an 3D image stored in slices in the DICOM file format.

    Parameters:
        image_orientation (list): 
            a list of 6 elements representing the ImageOrientationPatient DICOM tag for the image. 
            This specifies the orientation of the image slices in 3D space.
        image_positions (list): 
            a list of 3-element lists representing the ImagePositionPatient DICOM tag for each slice in the image. 
            This specifies the position of each slice in 3D space.
        pixel_spacing (list): 
            a list of 2 elements representing the PixelSpacing DICOM tag for the image. 
            This specifies the spacing between pixels in the rows and columns of each slice.
        rows (int): 
            an integer representing the number of rows in each slice.
        columns (int): 
            an integer representing the number of columns in each slice.

    Returns:
        dict: a dictionary with keys 'RPF', 'LPF', 'LPH', 'RPH', 'RAF', 'LAF', 'LAH', and 'RAH', 
        representing the Right Posterior Foot, Left Posterior Foot, Left Posterior Head, 
        Right Posterior Head, Right Anterior Foot, Left Anterior Foot, 
        Left Anterior Head, and Right Anterior Head, respectively. 
        Each key maps to a list of 3 elements representing the x, y, and z coordinates 
        of the corresponding corner of the bounding box.
   
    """

    row_spacing = pixel_spacing[0]
    column_spacing = pixel_spacing[1]

    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)

    number_of_slices = len(image_positions)
    image_locations = [np.dot(np.array(pos), slice_cosine) for pos in image_positions]
    slab_thickness = max(image_locations) - min(image_locations)
    slice_spacing = slab_thickness / (number_of_slices - 1)
    image_position_first_slice = image_positions[image_locations.index(min(image_locations))]

    # ul = Upper Left corner of a slice
    # ur = Upper Right corner of a slice
    # bl = Bottom Left corner of a slice
    # br = Bottom Right corner of a slice
    
    # Initialize with the first slice
    ul = image_position_first_slice
    ur = ul + row_cosine * (columns-1) * column_spacing
    br = ur + column_cosine * (rows-1) * row_spacing
    bl = ul + column_cosine * (rows-1) * row_spacing
    corners = np.array([ul, ur, br, bl])
    amin = np.amax(corners, axis=0)
    amax = np.amax(corners, axis=0)
    box = {
        'RPF': [amin[0],amax[1],amin[2]], # Right Posterior Foot 
        'LPF': [amax[0],amax[1],amin[2]], # Left Posterior Foot
        'LPH': [amax[0],amax[1],amax[2]], # Left Posterior Head
        'RPH': [amin[0],amax[1],amax[2]], # Right Posterior Head
        'RAF': [amin[0],amin[1],amin[2]], # Right Anterior Foot
        'LAF': [amax[0],amin[1],amin[2]], # Left Anterior Foot
        'LAH': [amax[0],amin[1],amax[2]], # Left Anterior Head
        'RAH': [amin[0],amin[1],amax[2]], # Right Anterior Head
    }

    # Update with all other slices
    # PROBABLY SUFFICIENT TO USE ONLY THE OUTER SLICES!!
    for _ in range(1, number_of_slices):

        ul += slice_cosine * slice_spacing
        ur = ul + row_cosine * (columns-1) * column_spacing
        br = ur + column_cosine * (rows-1) * row_spacing
        bl = ul + column_cosine * (rows-1) * row_spacing

        corners = np.array([ul, ur, br, bl])
        amin = np.amin(corners, axis=0)
        amax = np.amax(corners, axis=0)

        box['RPF'][0] = min([box['RPF'][0], amin[0]])    
        box['RPF'][1] = max([box['RPF'][1], amax[1]]) 
        box['RPF'][2] = min([box['RPF'][2], amin[2]]) 

        box['LPF'][0] = max([box['LPF'][0], amax[0]]) 
        box['LPF'][1] = max([box['LPF'][1], amax[1]]) 
        box['LPF'][2] = min([box['LPF'][2], amin[2]]) 

        box['LPH'][0] = max([box['LPH'][0], amax[0]]) 
        box['LPH'][1] = max([box['LPH'][1], amax[1]]) 
        box['LPH'][2] = max([box['LPH'][2], amax[2]]) 

        box['RPH'][0] = min([box['RPH'][0], amin[0]]) 
        box['RPH'][1] = max([box['RPH'][1], amax[1]]) 
        box['RPH'][2] = max([box['RPH'][2], amax[2]]) 

        box['RAF'][0] = min([box['RAF'][0], amin[0]]) 
        box['RAF'][1] = min([box['RAF'][1], amin[1]]) 
        box['RAF'][2] = min([box['RAF'][2], amin[2]]) 

        box['LAF'][0] = max([box['LAF'][0], amax[0]]) 
        box['LAF'][1] = min([box['LAF'][1], amin[1]]) 
        box['LAF'][2] = min([box['LAF'][2], amin[2]]) 

        box['LAH'][0] = max([box['LAH'][0], amax[0]]) 
        box['LAH'][1] = min([box['LAH'][1], amin[1]]) 
        box['LAH'][2] = max([box['LAH'][2], amax[2]]) 

        box['RAH'][0] = min([box['RAH'][0], amin[0]]) 
        box['RAH'][1] = min([box['RAH'][1], amin[1]]) 
        box['RAH'][2] = max([box['RAH'][2], amax[2]]) 

    return box



def standard_affine_matrix(
    bounding_box, 
    pixel_spacing, 
    slice_spacing,
    orientation = 'axial'): 

    row_spacing = pixel_spacing[0]
    column_spacing = pixel_spacing[1]
    
    if orientation == 'axial':
        image_position = bounding_box['RAF']
        row_cosine = np.array([1,0,0])
        column_cosine = np.array([0,1,0])
        slice_cosine = np.array([0,0,1])
    elif orientation == 'coronal':
        image_position = bounding_box['RAH']
        row_cosine = np.array([1,0,0])
        column_cosine = np.array([0,0,-1])
        slice_cosine = np.array([0,1,0]) 
    elif orientation == 'sagittal':
        image_position = bounding_box['LAH']
        row_cosine = np.array([0,1,0])
        column_cosine = np.array([0,0,-1])
        slice_cosine = np.array([-1,0,0])          

    affine = np.identity(4, dtype=np.float32)
    affine[:3, 0] = row_cosine * column_spacing
    affine[:3, 1] = column_cosine * row_spacing
    affine[:3, 2] = slice_cosine * slice_spacing
    affine[:3, 3] = image_position
    
    return affine 


def affine_matrix(      # single slice function
    image_orientation,  # ImageOrientationPatient
    image_position,     # ImagePositionPatient
    pixel_spacing,      # PixelSpacing
    slice_spacing):     # SpacingBetweenSlices
    """
    Calculate an affine transformation matrix for a single slice of an image in the DICOM file format.
    The affine transformation matrix can be used to transform the image from its original coordinates
    to a new set of coordinates.

    Parameters:
        image_orientation (list): a list of 6 elements representing the ImageOrientationPatient
                                  DICOM tag for the image. This specifies the orientation of the
                                  image slices in 3D space.
        image_position (list): a list of 3 elements representing the ImagePositionPatient DICOM
                               tag for the slice. This specifies the position of the slice in 3D space.
        pixel_spacing (list): a list of 2 elements representing the PixelSpacing DICOM tag for the
                              image. This specifies the spacing between pixels in the rows and columns
                              of each slice.
        slice_spacing (float): a float representing the SpacingBetweenSlices DICOM tag for the image. This
                               specifies the spacing between slices in the image.

    Returns:
        np.ndarray: an affine transformation matrix represented as a 4x4 NumPy array with dtype `float32`.
                    The matrix can be used to transform the image from its original coordinates to a new set
                    of coordinates.
    """

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
    
    return affine 



def affine_matrix_multislice(
    image_orientation,  # ImageOrientationPatient (assume same for all slices)
    image_positions,    # ImagePositionPatient for all slices
    pixel_spacing):     # PixelSpacing (assume same for all slices)

    row_spacing = pixel_spacing[0]
    column_spacing = pixel_spacing[1]
    
    row_cosine = np.array(image_orientation[:3])    
    column_cosine = np.array(image_orientation[3:]) 
    slice_cosine = np.cross(row_cosine, column_cosine)

    image_locations = [np.dot(np.array(pos), slice_cosine) for pos in image_positions]
    #number_of_slices = len(image_positions)
    number_of_slices = np.unique(image_locations).size
    if number_of_slices == 1:
        msg = 'Cannot calculate affine matrix for the slice group. \n'
        msg += 'All slices have the same location. \n'
        msg += 'Use the single-slice affine formula instead.'
        raise ValueError(msg)
    slab_thickness = np.amax(image_locations) - np.amin(image_locations)
    slice_spacing = slab_thickness / (number_of_slices - 1)
    image_position_first_slice = image_positions[image_locations.index(np.amin(image_locations))]

    affine = np.identity(4, dtype=np.float32)
    affine[:3, 0] = row_cosine * column_spacing 
    affine[:3, 1] = column_cosine * row_spacing
    affine[:3, 2] = slice_cosine * slice_spacing
    affine[:3, 3] = image_position_first_slice

    return affine


def dismantle_affine_matrix(affine):
    # Note: nr of slices can not be retrieved from affine_matrix
    # Note: slice_cosine is not a DICOM keyword but can be used 
    # to work out the ImagePositionPatient of any other slice i as
    # ImagePositionPatient_i = ImagePositionPatient + i * SpacingBetweenSlices * slice_cosine
    column_spacing = np.linalg.norm(affine[:3, 0])
    row_spacing = np.linalg.norm(affine[:3, 1])
    slice_spacing = np.linalg.norm(affine[:3, 2])
    row_cosine = affine[:3, 0] / column_spacing
    column_cosine = affine[:3, 1] / row_spacing
    slice_cosine = affine[:3, 2] / slice_spacing
    return {
        'PixelSpacing': [row_spacing, column_spacing], 
        'SpacingBetweenSlices': slice_spacing,  # This is really spacing between slices
        'ImageOrientationPatient': row_cosine.tolist() + column_cosine.tolist(), 
        'ImagePositionPatient': affine[:3, 3].tolist(), # first slice for a volume
        'slice_cosine': slice_cosine.tolist()} 

def affine_to_RAH(affine):
    """Convert to the coordinate system used in NifTi"""

    rot_180 = np.identity(4, dtype=np.float32)
    rot_180[:2,:2] = [[-1,0],[0,-1]]
    return np.matmul(rot_180, affine)
    

def image_position_patient(affine, number_of_slices):
    slab = dismantle_affine_matrix(affine)
    image_positions = []
    image_locations = []
    for s in range(number_of_slices):
        pos = [
            slab['ImagePositionPatient'][i] 
            + s*slab['SpacingBetweenSlices']*slab['slice_cosine'][i]
            for i in range(3)
        ]
        loc = np.dot(np.array(pos), np.array(slab['slice_cosine']))
        image_positions.append(pos)
        image_locations.append(loc)
    return image_positions, image_locations


def clip(array, value_range = None):

    array[np.isnan(array)] = 0
    if value_range is None:
        finite = array[np.isfinite(array)]
        value_range = [np.amin(finite), np.amax(finite)]
    return np.clip(array, value_range[0], value_range[1])
    

def scale_to_range(array, bits_allocated):
        
    range = 2.0**bits_allocated - 1
    maximum = np.amax(array)
    minimum = np.amin(array)
    if maximum == minimum:
        slope = 1
    else:
        slope = range / (maximum - minimum)
    intercept = -slope * minimum
    array *= slope
    array += intercept

    if bits_allocated == 8:
        return array.astype(np.uint8), slope, intercept
    if bits_allocated == 16:
        return array.astype(np.uint16), slope, intercept
    if bits_allocated == 32:
        return array.astype(np.uint32), slope, intercept
    if bits_allocated == 64:
        return array.astype(np.uint64), slope, intercept


def BGRA(array, RGBlut=None, width=None, center=None):

    if (width is None) or (center is None):
        max = np.amax(array)
        min = np.amin(array)
    else:
        max = center+width/2
        min = center-width/2

    # Scale pixel array into byte range
    array = np.clip(array, min, max)
    array -= min
    if max > min:
        array *= 255/(max-min)
    array = array.astype(np.ubyte)

    BGRA = np.empty(array.shape[:2]+(4,), dtype=np.ubyte)
    BGRA[:,:,3] = 255 # Alpha channel

    if RGBlut is None:
        # Greyscale image
        for c in range(3):
            BGRA[:,:,c] = array
    else:
        # Scale LUT into byte range
        RGBlut *= 255
        RGBlut = RGBlut.astype(np.ubyte)       
        # Create RGB array by indexing LUT with pixel array
        for c in range(3):
            BGRA[:,:,c] = RGBlut[array,2-c]

    return BGRA




