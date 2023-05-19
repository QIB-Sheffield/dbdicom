import time
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
import sklearn

import pyvista as pv

from skimage.draw import ellipsoid


##########################
# Helper functions
##########################


def volume_coordinates(shape, position=[0,0,0]):

    # data are defined at the middle of the voxels - use p+0.5 offset.
    xo, yo, zo = np.meshgrid(
        np.arange(position[0]+0.5, position[0]+0.5+shape[0]),
        np.arange(position[1]+0.5, position[1]+0.5+shape[1]),
        np.arange(position[2]+0.5, position[2]+0.5+shape[2]),
        indexing = 'ij')
    return np.column_stack((xo.ravel(), yo.ravel(), zo.ravel()))


def interpolate_displacement(displacement_field, shape, **kwargs):

    # Get the coordinates of the displacement field with dimensions (x,y,z,d)
    w = np.array(displacement_field.shape[:-1])-1
    d = np.divide(w, shape)
    xo, yo, zo = np.meshgrid(
        np.linspace(0.5*d[0], w[0]-0.5*d[0], shape[0]),
        np.linspace(0.5*d[1], w[1]-0.5*d[1], shape[1]),
        np.linspace(0.5*d[2], w[2]-0.5*d[2], shape[2]),
        indexing = 'ij')
    co = np.column_stack((xo.ravel(), yo.ravel(), zo.ravel()))

    # Interpolate displacement field in volume coordinates
    #deformation = ndi.map_coordinates(displacement_field, co.T, **kwargs)
    deformation = np.column_stack(
        (
            ndi.map_coordinates(displacement_field[...,0], co.T, **kwargs),
            ndi.map_coordinates(displacement_field[...,1], co.T, **kwargs),
            ndi.map_coordinates(displacement_field[...,2], co.T, **kwargs),
        )
    )
    #deformation = np.reshape(deformation, shape+(3,))
    return deformation.reshape((np.prod(shape), 3))


def surface_coordinates(shape):

    # data are defined at the edge of volume - extend shape with 1.
    xo, yo, zo = np.meshgrid(
        np.arange(1.0 + shape[0]),
        np.arange(1.0 + shape[1]),
        np.arange(1.0 + shape[2]),
        indexing = 'ij')
    return np.column_stack((xo.ravel(), yo.ravel(), zo.ravel()))


def extend_border(r, shape):

    # Shift with half a voxel because data are defined at voxel centers
    r -= 0.5

    # Set coordinates at 0.5 pixel from the borders equal to the borders
    x0, x1 = 0, shape[0]-1
    y0, y1 = 0, shape[1]-1
    z0, z1 = 0, shape[2]-1

    r[np.where(np.logical_and(x0-0.5 <= r[:,0], r[:,0] <= x0)), 0] = x0
    r[np.where(np.logical_and(x1+0.5 >= r[:,0], r[:,0] >= x1)), 0] = x1
    r[np.where(np.logical_and(y0-0.5 <= r[:,1], r[:,1] <= y0)), 1] = y0
    r[np.where(np.logical_and(y1+0.5 >= r[:,1], r[:,1] >= y1)), 1] = y1
    r[np.where(np.logical_and(z0-0.5 <= r[:,2], r[:,2] <= z0)), 2] = z0
    r[np.where(np.logical_and(z1+0.5 >= r[:,2], r[:,2] >= z1)), 2] = z1

    return r


def pv_contour(values, data, affine, surface=False):

    # For display of the surface, interpolate from volume to surface array
    surf_shape = 1 + np.array(data.shape)
    r = surface_coordinates(data.shape)
    r = extend_border(r, data.shape)
    surf_data = ndi.map_coordinates(data, r.T, order=3)
    surf_data = np.reshape(surf_data, surf_shape)

    rotation, translation, pixel_spacing = affine_components(affine)
    grid = pv.UniformGrid(dimensions=surf_shape, spacing=pixel_spacing)
    surf = grid.contour(values, surf_data.flatten(order="F"), method='marching_cubes')
    surf = surf.rotate_vector(rotation, np.linalg.norm(rotation)*180/np.pi, inplace=False)
    surf = surf.translate(translation, inplace=False)
    if surface:
        surf = surf.reconstruct_surface()
    return surf


def parallellepid(L, affine=None):

    c = np.array([0,0,0])
    x = np.array([1,0,0])*L[0]
    y = np.array([0,1,0])*L[1]
    z = np.array([0,0,1])*L[2]

    # mesh points
    vertices = np.array(
        [   c, 
            c+x, 
            c+x+z, 
            c+z, 
            c+y, 
            c+y+x, 
            c+y+x+z, 
            c+y+z,
        ]
    )
    
    if affine is not None:
        nd = 3
        matrix = affine[:nd,:nd]
        offset = affine[:nd, nd]
        vertices = np.dot(vertices, matrix.T) + offset

    # mesh faces
    faces = np.hstack(
        [
            [4, 0, 1, 2, 3], #right
            [4, 4, 5, 6, 7], #left
            [4, 0, 1, 5, 4], #bottom
            [4, 2, 3, 7, 6], #top
            [4, 0, 3, 7, 4], #front
            [4, 1, 2, 6, 5], #back
        ]
    )

    return vertices, faces


def rotation_displacement(rotation, center):

    rot = Rotation.from_rotvec(rotation)
    center_rot = rot.apply(center)
    return center_rot-center


def center_of_mass(volume, affine):

    center_of_mass = ndi.center_of_mass(volume)
    nd = volume.ndim
    matrix = affine[:nd,:nd]
    offset = affine[:nd, nd]
    return np.dot(matrix, center_of_mass) + offset


def affine_components(matrix):
    """Extract rotation, translation and pixel spacing vector from affine matrix"""

    nd = matrix.shape[0]-1
    translation = matrix[:nd, nd].copy()
    rotation_matrix = matrix[:nd, :nd].copy()
    pixel_spacing = np.linalg.norm(matrix[:nd, :nd], axis=0)
    for c in range(nd):
        rotation_matrix[:nd, c] /= pixel_spacing[c]
    rot = Rotation.from_matrix(rotation_matrix)
    rotation = rot.as_rotvec()
    return rotation, translation, pixel_spacing


def affine_resolution(shape, spacing):
    """Smallest detectable rotation, translation and stretching of a volume with given shape and resolution."""

    translation_res = spacing
    rot_res_x = min([spacing[1],spacing[2]])/max([shape[1],shape[2]])
    rot_res_y = min([spacing[2],spacing[0]])/max([shape[2],shape[0]])
    rot_res_z = min([spacing[0],spacing[1]])/max([shape[0],shape[1]])
    rot_res = np.array([rot_res_x, rot_res_y, rot_res_z])
    scaling_res = np.array([0.01, 0.01, 0.01])
    return rot_res, translation_res, scaling_res


def affine_matrix(rotation=None, translation=None, pixel_spacing=None, center=None):

    nd = 3
    matrix = np.eye(1+nd)

    if rotation is not None:
        rot = Rotation.from_rotvec(rotation)
        matrix[:nd,:nd] = rot.as_matrix()

        # Shift to rotate around center
        if center is not None:
            center_rot = rot.apply(center)
            offset = center_rot-center
            matrix[:nd, nd] -= offset

    if translation is not None:
        matrix[:nd, nd] += translation

    if pixel_spacing is not None:
        for c in range(nd):
            matrix[:nd, c] *= pixel_spacing[c]

    return matrix


def envelope(d, affine):

    corners, _ = parallellepid(np.array(d), affine)

    x0 = np.amin(corners[:,0])
    x1 = np.amax(corners[:,0])
    y0 = np.amin(corners[:,1])
    y1 = np.amax(corners[:,1])
    z0 = np.amin(corners[:,2])
    z1 = np.amax(corners[:,2])

    nx = np.ceil(x1-x0).astype(np.int16)
    ny = np.ceil(y1-y0).astype(np.int16)
    nz = np.ceil(z1-z0).astype(np.int16)

    output_shape = (nx, ny, nz)
    output_pos = [x0, y0, z0]

    return output_shape, output_pos



def apply_affine(affine, coord):
    """Apply affine transformation to an array of coordinates"""

    nd = affine.shape[0]-1
    matrix = affine[:nd,:nd]
    offset = affine[:nd, nd]
    return np.dot(coord, matrix.T) + offset
    #return np.dot(matrix, co.T).T + offset


def affine_to_freeform(affine, output_shape, deformation_field_shape):
    xo, yo, zo = np.meshgrid(
        np.linspace(0, output_shape[0], deformation_field_shape[0]),
        np.linspace(0, output_shape[1], deformation_field_shape[1]),
        np.linspace(0, output_shape[2], deformation_field_shape[2]),
        indexing = 'ij')
    coordinates = np.column_stack((xo.ravel(), yo.ravel(), zo.ravel()))
    new_coordinates = apply_affine(affine, coordinates)
    deformation_field = new_coordinates - coordinates
    deformation_field = np.reshape(deformation_field, deformation_field_shape + (3,))
    return deformation_field


def apply_inverse_affine(input_data, inverse_affine, output_shape, output_coordinates=None, **kwargs):

    # Create an array of all coordinates in the output volume
    if output_coordinates is None:
        output_coordinates = volume_coordinates(output_shape)

    # Apply affine transformation to all coordinates in the output volume
    # nd = inverse_affine.shape[0]-1
    # matrix = inverse_affine[:nd,:nd]
    # offset = inverse_affine[:nd, nd]
    # input_coordinates = np.dot(output_coordinates, matrix.T) + offset
    # #co = np.dot(matrix, co.T).T + offset
    input_coordinates = apply_affine(inverse_affine, output_coordinates)

    # Extend with constant value for half a voxel outside of the boundary
    input_coordinates = extend_border(input_coordinates, input_data.shape)

    # Interpolate the volume in the transformed coordinates
    output_data = ndi.map_coordinates(input_data, input_coordinates.T, **kwargs)
    output_data = np.reshape(output_data, output_shape)

    return output_data


def affine_output_geometry(input_shape, input_affine, transformation):
        
    # Determine output shape and position
    affine_transformed = transformation.dot(input_affine)
    forward = np.linalg.inv(input_affine).dot(affine_transformed) # Ai T A
    output_shape, output_pos = envelope(input_shape, forward)

    # Determine output affine by shifting affine to the output position
    nd = input_affine.shape[0]-1
    matrix = input_affine[:nd,:nd]
    offset = input_affine[:nd, nd]
    output_affine = input_affine.copy()
    output_affine[:nd, nd] = np.dot(matrix, output_pos) + offset

    return output_shape, output_affine


####################################
## Affine transformation and reslice
####################################


# TODO This needs to become a private helper function
def affine_transform(input_data, input_affine, transformation, reshape=False, **kwargs):

    if reshape:
        output_shape, output_affine = affine_output_geometry(input_data.shape, input_affine, transformation)
    else:
        output_shape, output_affine = input_data.shape, input_affine.copy()

    # Perform the inverse transformation
    affine_transformed = transformation.dot(input_affine)
    inverse = np.linalg.inv(affine_transformed).dot(output_affine) # Ainv Tinv B 
    output_data = apply_inverse_affine(input_data, inverse, output_shape, **kwargs)

    return output_data, output_affine


# TODO This needs to become a private helper function
def affine_reslice(input_data, input_affine, output_affine, output_shape=None, **kwargs):

    # If no output shape is provided, retain the physical volume of the input datas
    if output_shape is None:

        # Get field of view from input data
        _, _, input_pixel_spacing = affine_components(input_affine)
        field_of_view = np.multiply(np.array(input_data.shape), input_pixel_spacing)

        # Find output shape for the same field of view
        output_rotation, output_translation, output_pixel_spacing = affine_components(output_affine)
        output_shape = np.around(np.divide(field_of_view, output_pixel_spacing)).astype(np.int16)
        output_shape[np.where(output_shape==0)] = 1

        # Adjust output pixel spacing to fit the field of view
        output_pixel_spacing = np.divide(field_of_view, output_shape)
        output_affine = affine_matrix(rotation=output_rotation, translation=output_translation, pixel_spacing=output_pixel_spacing)

    # Reslice input data to output geometry
    transform = np.linalg.inv(input_affine).dot(output_affine) # Ai B
    output_data = apply_inverse_affine(input_data, transform, output_shape, **kwargs)

    return output_data, output_affine


# This needs a reshape option to expand to the envelope in the new reference frame
def affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs):

    affine_transformed = transformation.dot(input_affine)
    inverse = np.linalg.inv(affine_transformed).dot(output_affine) # Ai Ti B
    output_data = apply_inverse_affine(input_data, inverse, output_shape, **kwargs)

    return output_data


# Deformation define in absolute coordinates
def absolute_freeform(input_data, input_affine, output_shape, output_affine, displacement, output_coordinates=None, **kwargs):

    # Create an array of all coordinates in the output volume
    if output_coordinates is None:
        output_coordinates = volume_coordinates(output_shape) 

    # Express output coordinates in the scanner reference frame
    reference_output_coordinates = apply_affine(output_affine, output_coordinates)

    # Apply free-from deformation to all output coordinates
    deformation = interpolate_displacement(displacement, output_shape)
    reference_input_coordinates = reference_output_coordinates + deformation

    # Express new coordinates in reference frame of the input volume
    input_affine_inv = np.linalg.inv(input_affine)
    input_coordinates = apply_affine(input_affine_inv, reference_input_coordinates)

    # Extend with constant value for half a voxel outside of the boundary
    # TODO make this an option - costs time and is not necessary when a window is taken inside the FOV (better way to deal with borders)
    input_coordinates = extend_border(input_coordinates, input_data.shape)

    # Interpolate the input data in the transformed coordinates
    output_data = ndi.map_coordinates(input_data, input_coordinates.T, **kwargs)
    output_data = np.reshape(output_data, output_shape)

    return output_data

# Note: to apply to a window, adjust output_shape and output_affine (position vector)
# Deformation defined in input coordinates
def freeform(input_data, input_affine, output_shape, output_affine, displacement, output_coordinates=None, **kwargs):

    # Create an array of all coordinates in the output volume
    # Optional argument as this can be precomputed for registration purposes
    if output_coordinates is None:
        output_coordinates = volume_coordinates(output_shape)
        # Express output coordinates in reference frame of the input volume
        transform = np.linalg.inv(input_affine).dot(output_affine)
        output_coordinates = apply_affine(transform, output_coordinates)
        
    # Apply free-from deformation to all output coordinates
    deformation = interpolate_displacement(displacement, output_shape)
    input_coordinates = output_coordinates + deformation

    # Extend with constant value for half a voxel outside of the boundary
    # TODO make this an option in 3D - costs time and is not necessary when a window is taken inside the FOV (better way to deal with borders)
    input_coordinates = extend_border(input_coordinates, input_data.shape)

    # Interpolate the input data in the transformed coordinates
    output_data = ndi.map_coordinates(input_data, input_coordinates.T, **kwargs)
    output_data = np.reshape(output_data, output_shape)

    return output_data



####################################
# wrappers for use in align function
####################################



def translate(input_data, input_affine, output_shape, output_affine, translation, **kwargs):
    transformation = affine_matrix(translation=translation)
    return affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def translate_reshape(input_data, input_affine, translation, **kwargs):
    transformation = affine_matrix(translation=translation)
    return affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def rotate(input_data, input_affine, output_shape, output_affine, rotation, **kwargs):
    transformation = affine_matrix(rotation=rotation)
    return affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rotate_reshape(input_data, input_affine, rotation, **kwargs):
    transformation = affine_matrix(rotation=rotation)
    return affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def stretch(input_data, input_affine, output_shape, output_affine, stretch, **kwargs):
    transformation = affine_matrix(pixel_spacing=stretch)
    return affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def stretch_reshape(input_data, input_affine, stretch, **kwargs):
    transformation = affine_matrix(pixel_spacing=stretch)
    return affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def rotate_around(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = affine_matrix(rotation=parameters[:3], center=parameters[3:])
    return affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rotate_around_reshape(input_data, input_affine, rotation, center, **kwargs):
    transformation = affine_matrix(rotation=rotation, center=center)
    return affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def rigid(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = affine_matrix(rotation=parameters[:3], translation=parameters[3:])
    return affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def rigid_reshape(input_data, input_affine, rotation, translation, **kwargs):
    transformation = affine_matrix(rotation=rotation, translation=translation)
    return affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)

def affine(input_data, input_affine, output_shape, output_affine, parameters, **kwargs):
    transformation = affine_matrix(rotation=parameters[:3], translation=parameters[3:6], pixel_spacing=parameters[6:])
    return affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation, **kwargs)

def affine_reshape(input_data, input_affine, rotation, translation, stretch, **kwargs):
    transformation = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
    return affine_transform(input_data, input_affine, transformation, reshape=True, **kwargs)




# default metrics
# ---------------

def sum_of_squares(static, transformed, nan=None):
    if nan is not None:
        i = np.where(transformed != nan)
        return np.sum(np.square(static[i]-transformed[i]))
    else:
        return np.sum(np.square(static-transformed))


def mutual_information(static, transformed, nan=None):
    if nan is not None:
        i = np.where(transformed != nan)
        return sklearn.metrics.mutual_info_score(static[i], transformed[i])
    else:
        return sklearn.metrics.mutual_info_score(static, transformed)
    

def normalized_mutual_information(static, transformed, nan=None):
    if nan is not None:
        i = np.where(transformed != nan)
        return sklearn.metrics.normalized_mutual_info_score(static[i], transformed[i])
    else:
        return sklearn.metrics.normalized_mutual_info_score(static, transformed)



# Generic objective function
def _objective_function(
        params, 
        moving: np.ndarray, 
        static: np.ndarray, 
        transformation, 
        metric,
        moving_affine,
        static_affine,
        moving_mask, 
        static_mask, 
        transformation_args, 
        metric_args):

    # # Get static size (TODO: use shape instead of size in transformations)
    # _, _, static_pixel_spacing = affine_components(static_affine)
    # static_size = np.multiply(np.array(static.shape)-1, static_pixel_spacing)

    # Transform the moving image
    nan = 0
    if transformation_args is None:
        transformed = transformation(moving, moving_affine, static.shape, static_affine, params, cval=nan)
    else:
        transformed = transformation(moving, moving_affine, static.shape, static_affine, params, transformation_args, cval=nan)

    # If a moving mask is provided, transform this as well
    if moving_mask is not None:
        if transformation_args is None:
            transformed_mask = transformation(moving_mask, moving_affine, static.shape, static_affine, params)
        else:
            transformed_mask = transformation(moving_mask, moving_affine, static.shape, static_affine, params, transformation_args)
        transformed_mask[transformed_mask > 0.5] = 1
        transformed_mask[transformed_mask <= 0.5] = 0

    # Get the indices in the static data where the mask is non-zero
    if static_mask is not None and moving_mask is not None:
        ind = np.where(transformed!=nan and static_mask==1 and transformed_mask==1)
    elif static_mask is not None:
        ind = np.where(transformed!=nan and static_mask==1) 
    elif moving_mask is not None:
        ind = np.where(transformed!=nan and transformed_mask==1)
    else:
        ind = np.where(transformed!=nan)
        
    # Calculate the cost function         
    if metric_args is None:
        return metric(static[ind], transformed[ind])
    else:
        return metric(static[ind], transformed[ind], metric_args)


def print_current(xk):
    print('Current parameter estimates: ' , xk)
    return False


def minimize(*args, method='GD', **kwargs):
    "Wrapper around opt.minimize which also has a gradient descent option"

    if method == 'GD':
        return minimize_gd(*args, **kwargs)
    else:
        res = opt.minimize(*args, method=method, **kwargs)
        return res.x
    

def minimize_gd(cost_function, parameters, args=None, callback=None, options={}, bounds=None):

    # Set default values for global options
    if 'max_iter' in options:
        max_iter = options['max_iter']
    else:
        max_iter = 100 
    if 'gradient step' in options:
        step = options['gradient step']
    else:
        step = np.ones(parameters.shape)

    # set defaults for line search options
    ls_pars = ['tolerance', 'scale_down', 'scale_up', 'stepsize_max'] 
    ls_options = dict((k, options[k]) for k in ls_pars if k in options)

    stepsize = 1.0 # initial stepsize
    n_iter = 0
    cost = cost_function(parameters, *args)
    while True:
        n_iter+=1
        print('iteration: ', n_iter)
        grad = gradient(cost_function, parameters, cost, step, bounds, *args)
        parameters, stepsize, cost = line_search(cost_function, grad, parameters, stepsize, cost, bounds, *args, **ls_options)
        if callback is not None:
            callback(parameters)
        if cost == 0:
            return parameters
        if stepsize == 0:
            return parameters
        if n_iter == max_iter:
            return parameters


def _ONESIDED_gradient(cost_function, parameters, f0, step, *args):
    grad = np.empty(parameters.shape)
    for i, p in enumerate(parameters):
        pi = parameters[i]
        parameters[i] = pi+step[i]
        fi = cost_function(parameters, *args)
        parameters[i] = pi
        #grad[i] = (fi-f0)/step[i]
        grad[i] = (fi-f0)
        parameters[i] = parameters[i]-step[i]
    return grad


def gradient(cost_function, parameters, f0, step, bounds, *args):
    grad = np.empty(parameters.shape)
    for i, p in enumerate(parameters):
        pi = parameters[i]
        parameters[i] = pi+step[i]
        parameters = project_on(parameters, bounds, index=i)
        fp = cost_function(parameters, *args)
        parameters[i] = pi-step[i]
        parameters = project_on(parameters, bounds, index=i)
        fn = cost_function(parameters, *args)
        parameters[i] = pi
        grad[i] = (fp-fn)/2
        #grad[i] = (fp-fn)/(2*step[i]) 
        #grad[i] = stats.linregress([-step[i],0,step[i]], [fn,f0,fp]).slope

    # Normalize the gradient
    grad /= np.linalg.norm(grad)
    grad = np.multiply(step, grad)

    return grad


def project_on(par, bounds, index=None):
    if bounds is None:
        return par
    if len(bounds) != len(par):
        msg = 'Parameter and bounds must have the same length'
        raise ValueError(msg)
    if index is not None:   # project only that index
        pi = par[index]
        bi = bounds[index]
        if pi <= bi[0]:
            pi = bi[0]
        if pi >= bi[1]:
            pi = bi[1]
    else:   # project all indices
        for i, pi in enumerate(par):
            bi = bounds[i]
            if pi <= bi[0]:
                par[i] = bi[0]
            if pi >= bi[1]:
                par[i] = bi[1]
    return par


def line_search(cost_function, grad, p0, stepsize0, f0, bounds, *args, tolerance=0.1, scale_down=5.0, scale_up=1.5, stepsize_max=1000.0):

    # Initialize stepsize to current optimal stepsize
    stepsize_try = stepsize0 / scale_down
    p_init = p0.copy()

    # backtrack in big steps until reduction in cost
    while True:

        # Take a step and evaluate the cost
        p_try = p_init - stepsize_try*grad 
        p_try = project_on(p_try, bounds)
        f_try = cost_function(p_try, *args)

        print('cost: ', f_try, ' stepsize: ', stepsize_try, ' par: ', p_try)

        # If a reduction in cost is found, move on to the next part
        if f_try < f0:
            break

        # Otherwise reduce the stepsize and try again
        else:
            stepsize_try /= scale_down

        # If the stepsize has been reduced below the resolution without reducing the cost,
        # then the initial values were at the minimum (stepsize=0).
        if stepsize_try < tolerance: 
            return p0, 0, f0 # converged
        
    if stepsize_try < tolerance: 
        return p_try, 0, f_try # converged

    # If a reduction in cost has been found, then refine it 
    # by moving forward in babysteps until the cost increases again.
    while True:
        
        # Update the current optimum
        stepsize0 = stepsize_try
        f0 = f_try
        p0 = p_try

        # Take a baby step and evaluate the cost
        stepsize_try *= scale_up
        p_try = p_init - stepsize_try*grad
        p_try = project_on(p_try, bounds)
        f_try = cost_function(p_try, *args)

        print('cost: ', f_try, ' stepsize: ', stepsize_try, ' par: ', p_try)

        # If the cost has increased then a minimum was found
        if f_try >= f0:
            return p0, stepsize0, f0

        # emergency stop
        if stepsize_try > stepsize_max:
            msg = 'Line search failed to find a minimum'
            raise ValueError(msg) 
        

def _OLD_align(
        moving = None, 
        static = None, 
        parameters = np.zeros(3), 
        transformation = translate, 
        metric = sum_of_squares, 
        moving_affine = None,
        static_affine = None,
        moving_mask = None, 
        static_mask = None, 
        transformation_args = None, 
        metric_args = None):
    
    # Set defaults for required parameters
    if moving_affine is None:
        moving_affine = np.eye(moving.ndim + 1)
    if static_affine is None:
        static_affine = np.eye(static.ndim + 1)
    
    args = (
        moving, 
        static, 
        transformation, 
        metric, 
        moving_affine,
        static_affine,
        moving_mask, 
        static_mask, 
        transformation_args, 
        metric_args)
    
    res = opt.minimize(
        _objective_function, 
        parameters, 
        args=args,
        callback=_callback,
    )

    return res.x


def goodness_of_alignment(params, transformation, metric, moving, moving_affine, static, static_affine, coord):

    # Transform the moving image
    nan = 2**16-2 #np.nan does not work
    transformed = transformation(moving, moving_affine, static.shape, static_affine, params, output_coordinates=coord, cval=nan)
        
    # Calculate the cost function  
    ls = metric(static, transformed, nan=nan)
    
    return ls
    

def align(
        moving = None, 
        static = None, 
        parameters = None, 
        moving_affine = None, 
        static_affine = None, 
        transformation = translate,
        metric = sum_of_squares,
        optimization = {'method':'GD', 'options':{}},
        resolutions = [1]):
    
    # Set defaults
    if moving is None:
        msg = 'The moving volume is a required argument for alignment'
        raise ValueError(msg)
    if static is None:
        msg = 'The static volume is a required argument for alignment'
        raise ValueError(msg)
    if parameters is None:
        msg = 'Initial values for alignment must be provided'
        raise ValueError(msg)
    if moving_affine is None:
        moving_affine = np.eye(1 + moving.ndim)
    if static_affine is None:
        static_affine = np.eye(1 + static.ndim)

    # Perform multi-resolution loop
    for res in resolutions:
        print('DOWNSAMPLE BY FACTOR: ', res)

        if res == 1:
            moving_resampled, moving_resampled_affine = moving, moving_affine
            static_resampled, static_resampled_affine = static, static_affine
        else:
            # Downsample moving data
            r, t, p = affine_components(moving_affine)
            moving_resampled_affine = affine_matrix(rotation=r, translation=t, pixel_spacing=p*res)
            moving_resampled, moving_resampled_affine = affine_reslice(moving, moving_affine, moving_resampled_affine)
            #moving_resampled_data, moving_resampled_affine = moving_data, moving_affine

            # Downsample static data
            r, t, p = affine_components(static_affine)
            static_resampled_affine = affine_matrix(rotation=r, translation=t, pixel_spacing=p*res)
            static_resampled, static_resampled_affine = affine_reslice(static, static_affine, static_resampled_affine)

        coord = volume_coordinates(static_resampled.shape)
        args = (transformation, metric, moving_resampled, moving_resampled_affine, static_resampled, static_resampled_affine, coord)
        parameters = minimize(goodness_of_alignment, parameters, args=args, **optimization)

    return parameters


#############################
# Plot results
#############################


def plot_volume(volume, affine):

    clr, op = (255,255,255), 0.5
    #clr, op = (255,0,0), 1.0

    pl = pv.Plotter()
    pl.add_axes()

    # Plot the surface
    surf = pv_contour([0.5], volume, affine)
    if len(surf.points) == 0:
        print('Cannot plot the reference surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=clr, 
            opacity=op,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )

    # Plot wireframe around edges of reference volume
    vertices, faces = parallellepid(volume.shape, affine=affine)
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        opacity=op,
        color=clr, 
    )
    
    return pl


def plot_affine_resliced(volume, affine, volume_resliced, affine_resliced):

    clr, op = (255,0,0), 1.0

    pl = plot_volume(volume, affine)

    # Plot the resliced surface
    surf = pv_contour([0.5], volume_resliced, affine_resliced)
    if len(surf.points) == 0:
        print('Cannot plot the resliced surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=clr, 
            opacity=op,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )

    # Plot wireframe around edges of resliced volume
    vertices, faces = parallellepid(volume_resliced.shape, affine=affine_resliced)
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        opacity=op,
        color=clr, 
    ) 

    return pl


def plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation):

    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine)

    # Plot the reference surface
    surf = pv_contour([0.5], output_data, output_affine)
    if len(surf.points) == 0:
        print('Cannot plot the surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=(0,0,255), 
            opacity=0.25,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )
    
    # Create blue reference box showing transformation
    vertices, faces = parallellepid(input_data.shape, affine=np.dot(transformation, input_affine)) 
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        color=(0,0,255),
        opacity=0.5,
    )            
        
    pl.show()


def plot_freeform_transformed(input_data, input_affine, output_data, output_affine, transformation):

    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine)

    # Plot the reference surface
    surf = pv_contour([0.5], output_data, output_affine)
    if len(surf.points) == 0:
        print('Cannot plot the surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=(0,0,255), 
            opacity=0.25,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )           
        
    pl.show()


def plot_affine_transform_reslice(input_data, input_affine, output_data, output_affine, transformation):

    # Plot reslice
    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine)
    
    # Show in blue transparent the transformation without reslicing
    output_data, output_affine = affine_transform(input_data, input_affine, transformation, reshape=True)

    # Plot the reference surface
    surf = pv_contour([0.5], output_data, output_affine)
    if len(surf.points) == 0:
        print('Cannot plot the surface. It has no points inside the volume. ')
    else:
        pl.add_mesh(surf,
            color=(0,0,255), 
            opacity=0.25,
            show_edges=False, 
            smooth_shading=True, 
            specular=0, 
            show_scalar_bar=False,        
        )

    # Create blue reference box showing transformation
    vertices, faces = parallellepid(input_data.shape, affine=np.dot(transformation, input_affine)) # is this correct to take the product? (?need a function affine_compose(A,B))
    box = pv.PolyData(vertices, faces)
    pl.add_mesh(box,
        style='wireframe',
        color=(0,0,255),
        opacity=0.5,
    )  

    pl.show()


#############################
# Generate test data
#############################


def generate(structure='ellipsoid', shape=None, affine=None, markers=True):
    
    # Default shape
    if shape is None:
        shape = (256, 256, 40) 

    # Default affine
    if affine is None:
        pixel_spacing = np.array([1.5, 1.5, 5]) # mm
        translation = np.array([0, 0, 0]) # mm
        rotation_angle = -0.2 * (np.pi/2) # radians
        rotation_axis = [1,0,0]
        rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
        affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    _, _ , pixel_spacing = affine_components(affine) 
    data = np.zeros(shape, dtype=np.float32)  

    if markers:
        # Insert cube markers at corners
        marker_width = 20 # marker width in mm
        w = np.around(np.divide(np.array([marker_width]*3), pixel_spacing))
        w = w.astype(np.int16) 
        data[0:w[0],0:w[1],0:w[2]] = 1
        data[-w[0]:,0:w[1],0:w[2]] = 1
        data[0:w[0],-w[1]:,0:w[2]] = 1
        data[-w[0]:,-w[1]:,0:w[2]] = 1
        data[0:w[0],0:w[1],-w[2]:] = 1
        data[-w[0]:,0:w[1],-w[2]:] = 1
        data[0:w[0],-w[1]:,-w[2]:] = 1
        data[-w[0]:,-w[1]:,-w[2]:] = 1

    if structure == 'ellipsoid':
        half_length = (20, 30, 40) # mm
        ellip = ellipsoid(half_length[0], half_length[1], half_length[2], spacing=pixel_spacing, levelset=False)
        d = ellip.shape
        p = [30, 30, 10]
        data[p[0]:p[0]+d[0], p[1]:p[1]+d[1], p[2]:p[2]+d[2]] = ellip
        return data, affine
    
    elif structure == 'double ellipsoid':
        half_length1 = np.array([10, 20, 30]) # mm
        half_length2 = np.array([5, 10, 15]) # mm
        pos = np.array([150, 50, 0]) # mm
        ellip1 = ellipsoid(half_length1[0], half_length1[1], half_length1[2], spacing=pixel_spacing, levelset=False)
        ellip2 = ellipsoid(half_length2[0], half_length2[1], half_length2[2], spacing=pixel_spacing, levelset=False)
        ellip1 = ellip1.astype(np.int16)
        ellip2 = ellip2.astype(np.int16)

        p = np.around(np.divide(pos, pixel_spacing)).astype(np.int16)
        d = ellip1.shape
        data[p[0]:p[0]+d[0], p[1]:p[1]+d[1], p[2]:p[2]+d[2]] = ellip1

        p += np.around([d[0], d[1]/4, d[2]/2]).astype(np.int16)
        d = ellip2.shape
        data[p[0]:p[0]+d[0], p[1]:p[1]+d[1], p[2]:p[2]+d[2]] = ellip2

        return data, affine

    elif structure == 'triple ellipsoid':
        half_length1 = np.array([10, 20, 30]) # mm
        half_length2 = np.array([5, 10, 15]) # mm
        p1 = np.array([150, 50, 10]) # mm
        p2 = np.array([170, 70, 20]) # mm
        p3 = np.array([150, 150, 10]) # mm

        ellip1 = ellipsoid(half_length1[0], half_length1[1], half_length1[2], spacing=pixel_spacing, levelset=False)
        ellip2 = ellipsoid(half_length2[0], half_length2[1], half_length2[2], spacing=pixel_spacing, levelset=False)
        ellip1 = ellip1.astype(np.int16)
        ellip2 = ellip2.astype(np.int16)

        p = np.around(np.divide(p1, pixel_spacing)).astype(np.int16)
        d = ellip1.shape
        data[p[0]:p[0]+d[0], p[1]:p[1]+d[1], p[2]:p[2]+d[2]] = ellip1
        
        p = np.around(np.divide(p2, pixel_spacing)).astype(np.int16)
        d = ellip2.shape
        data[p[0]:p[0]+d[0], p[1]:p[1]+d[1], p[2]:p[2]+d[2]] = ellip2

        p = np.around(np.divide(p3, pixel_spacing)).astype(np.int16)
        d = ellip1.shape
        data[p[0]:p[0]+d[0], p[1]:p[1]+d[1], p[2]:p[2]+d[2]] = ellip1

        return data, affine 


def generate_plot_data_1():

    # Define geometry of input data
    pixel_spacing = np.array([2.0, 2.0, 10.0]) # mm
    input_shape = np.array([100, 100, 10], dtype=np.int16)
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    
    # Generate ground truth data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=True)

    return input_data, input_affine


def generate_reslice_data_1():

    # Downsample
    # Reslice high-res volume to lower resolution.
    # Values are chosen so that the field of view stays the same.

    # Define geometry of input data
    matrix = np.array([400, 300, 120])  
    pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = -0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)

    # Define geometry of output data
    output_pixel_spacing = np.array([1.25, 5.0, 10.0]) # mm
    output_shape = None # retain field of view

    # Generate data
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=output_pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=matrix, affine=input_affine, markers=True)

    return input_data, input_affine, output_affine, output_shape


def generate_reslice_data_2():

    # Upsample
    # Reslice low-res volume to higher resolution.
    # Values are chosen so that the field of view stays the same.

    # Define geometry of input data
    matrix = np.array([320, 60, 8])  
    pixel_spacing = np.array([1.25, 5.0, 15.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = -0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)

    # Define geometry of output data
    output_pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    output_shape = None # retain field of view

    # Generate data
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=output_pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=matrix, affine=input_affine, markers=False)

    return input_data, input_affine, output_affine, output_shape


def generate_reslice_data_3():

    # resample to lower resolution with a
    # 90 degree rotation around x + translation along y

    # Define source data
    matrix = np.array([400, 300, 120]) 
    pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]

    # Generate source data
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=matrix, affine=input_affine)

    # Define geometry of new slab
    pixel_spacing = np.array([1.25, 1.25, 10.0]) # mm
    translation = np.array([0, 120, 0]) # mm
    rotation_angle = 1.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    
    # Reslice current slab to geometry of new slab
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    output_shape = None
    return input_data, input_affine, output_affine, output_shape


def generate_reslice_data_4():

    # Rotation at low resolution

    # Define geometry of input data
    input_shape = np.array([40, 40, 20], dtype=np.int16)
    pixel_spacing = np.array([6, 6, 6.0])       # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define output affine
    #pixel_spacing = np.array([0.5, 0.5, 0.5])       # mm
    translation = np.array([0, 0, 0])     # mm
    rotation_angle = 0.15 * (np.pi/2)    # radians
    #translation = np.array([0, 0, 30]) # mm
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    
    # Generate input data data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=True)

    return input_data, input_affine, output_affine, None


def generate_reslice_data_5():

    # Reslice an object with its own affine

    # Define geometry of input data
    input_size = np.array([400, 300, 120])   # mm
    input_shape = np.array([400, 300, 120], dtype=np.int16)
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    pixel_spacing = np.divide(input_size, input_shape)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define output affine
    pixel_spacing = np.array([1.6, 2.6, 7.5])       # mm
    translation = np.array([100, 0, 0])     # mm
    rotation_angle = 0.1 * (np.pi/2)    # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    
    # Generate input data data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

    # Reslice to output affine keeping the same field of view
    output_data, output_affine = affine_reslice(input_data, input_affine, output_affine)

    return output_data, output_affine, output_affine, None


def generate_reslice_data_6():

    # 1-pixel thick - does not work yet!!

    # Define geometry of input data
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    input_shape = np.array([300, 200, 20], dtype=np.int16)
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define geometry of output data
    pixel_spacing = np.array([1.25, 1.25, 10.0]) # mm
    output_shape = np.array([300, 250, 1], dtype=np.int16)
    translation = np.array([0, 0, 5]) # mm
    rotation_angle = 0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    
    # Generate ground truth data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=True)

    return input_data, input_affine, output_affine, output_shape


def generate_translated_data_1():

    # Define geometry of input data
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.5 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_shape = np.array([300, 200, 25])  

    # Define affine transformation
    translation = np.array([10, -10, 10]) # mm
    
    # Define geometry of output data (exactly equal to translated volume)
    transformation = affine_matrix(translation=translation)
    output_shape, output_affine = affine_output_geometry(input_shape, input_affine, transformation)

    # Generate ground truth data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)
    output_data = translate(input_data, input_affine, output_shape, output_affine, translation)

    return input_data, input_affine, output_data, output_affine, translation


def generate_translated_data_2():

    # Model for 3D to 2D registration

    # Define geometry of input data
    pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_shape = np.array([300, 200, 100]) 

    # Define affine transformation
    active_translation = np.array([10, -10, 10]) # mm

    # Define geometry of output data
    output_shape = np.array([150, 200, 1])  
    pixel_spacing = np.array([1.25, 1.25, 7.5]) # mm
    #translation = np.array([100, 0, 50]) # mm
    translation = np.array([100, 0, 25]) # mm
    rotation_angle = 0.1 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Generate ground truth data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)
    output_data = translate(input_data, input_affine, output_shape, output_affine, active_translation)

    return input_data, input_affine, output_data, output_affine, active_translation


def generate_translated_data_3():

    # Model for 2D to 3D registration
    # Same as 2 but input and output reversed

    # Define geometry of input data
    pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_shape = np.array([300, 200, 100]) 

    # Define affine transformation
    active_translation = np.array([10, -10, 10]) # mm

    # Define geometry of output data
    output_shape = np.array([150, 200, 1])   # mm
    pixel_spacing = np.array([1.25, 1.25, 7.5]) # mm
    # translation = np.array([100, 0, 50]) # mm
    translation = np.array([100, 0, 25]) # mm
    rotation_angle = 0.1 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Generate ground truth data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)
    output_data = translate(input_data, input_affine, output_shape, output_affine, active_translation)

    return output_data, output_affine, input_data, input_affine, -active_translation


############################
#### Define tests
############################


def test_plot(n=1):

    if n==1:
        input_data, input_affine = generate_plot_data_1()

    pl = plot_volume(input_data, input_affine)
    pl.show_grid()
    pl.show()


def test_affine_reslice(n=1):

    if n==1:
        input_data, input_affine, output_affine, output_shape = generate_reslice_data_1()
    elif n==2:
        input_data, input_affine, output_affine, output_shape = generate_reslice_data_2()
    elif n==3:
        input_data, input_affine, output_affine, output_shape = generate_reslice_data_3()
    elif n==4:
        input_data, input_affine, output_affine, output_shape = generate_reslice_data_4()
    elif n==5:
        input_data, input_affine, output_affine, output_shape = generate_reslice_data_5()
    elif n==6:
        input_data, input_affine, output_affine, output_shape = generate_reslice_data_6()

    start_time = time.time()
    output_data, output_affine = affine_reslice(input_data, input_affine, output_affine, output_shape=output_shape)
    end_time = time.time()
    
    # Display results
    print('Computation time (sec): ', end_time-start_time)
    pl = plot_affine_resliced(input_data, input_affine, output_data, output_affine)
    pl.show_grid()
    pl.show()


def test_affine_transform():

    # Define geometry of source data
    input_shape = np.array([300, 250, 25])   # mm
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]

    # Generate source volume data
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine)

    # Define affine transformation
    stretch = [1.0, 1, 2.0]
    translation = np.array([0, 20, 0]) # mm
    rotation_angle = 0.20 * (np.pi/2)
    rotation_axis = [0,0,1]
    
    # Perform affine transformation
    start_time = time.time()
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    transformation = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
    output_data, output_affine = affine_transform(input_data, input_affine, transformation)
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)


def test_affine_transform_reshape():

    # Define geometry of source data
    input_shape = np.array([300, 250, 25])   # mm
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]

    # Generate source volume data
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine)

    # Define affine transformation
    stretch = [1.0, 1, 2.0]
    translation = np.array([0, 20, 0]) # mm
    rotation_angle = 0.20 * (np.pi/2)
    rotation_axis = [0,0,1]
    
    # Perform affine transformation
    start_time = time.time()
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    transformation = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
    output_data, output_affine = affine_transform(input_data, input_affine, transformation, reshape=True)
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)


def test_affine_transform_and_reslice():

    # Define geometry of input data
    input_shape = np.array([400, 300, 120])   # mm
    pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = -0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define geometry of output data
    output_shape = np.array([350, 300, 10])
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([100, -30, -40]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define affine transformation
    stretch = [1.25, 1, 1.0]
    translation = np.array([20, 0, 0]) # mm
    rotation_angle = 0.1 * (np.pi/2)
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    transformation = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)

    # Generate input data
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine)

    # Calculate affine transform
    output_data = affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation)
    
    # Display results
    plot_affine_transform_reslice(input_data, input_affine, output_data, output_affine, transformation)


def test_translate(n):

    if n==1:
        input_data, input_affine, output_data, output_affine, translation = generate_translated_data_1()
    elif n==2:
        input_data, input_affine, output_data, output_affine, translation = generate_translated_data_2()
    elif n==3:
        input_data, input_affine, output_data, output_affine, translation = generate_translated_data_3()

    transformation = affine_matrix(translation=translation)
    plot_affine_transform_reslice(input_data, input_affine, output_data, output_affine, transformation)


def test_translate_reshape():

    # Define geometry of source data
    input_shape = np.array([300, 250, 12])   # mm
    pixel_spacing = np.array([1.25, 1.25, 10.0]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    translation = np.array([0, 0, 0]) # mm

    # Generate reference volume
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine)

    # Perform translation with reshaping
    translation = np.array([0, -30, 0]) # mm
    start_time = time.time()
    ouput_data, output_affine = translate_reshape(input_data, input_affine, translation)
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    transformation = affine_matrix(translation=translation)
    plot_affine_transformed(input_data, input_affine, ouput_data, output_affine, transformation)


def test_rotate(show=True):

    # Generate reference volume
    input_data, input_affine = generate('triple ellipsoid', markers=False)

    # Define rotation
    angle = 0.1 * (np.pi/2)
    axis = [1,0,0]
    rotation = angle * np.array(axis)/np.linalg.norm(axis)

    # Define output_volume
    output_shape = input_data.shape
    output_affine = input_affine

    # Perform rotation
    start_time = time.time()
    output_data = rotate(input_data, input_affine, output_shape, output_affine, rotation) # specifying outputshape and affine should not be required
    end_time = time.time()

    # Display results
    if show is True:
        print('Computation time (sec): ', end_time-start_time)
        transformation = affine_matrix(rotation=rotation)
        plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)

    return input_data, input_affine, output_data, output_affine, rotation


def test_rotate_reshape():

    # Generate reference volume
    input_data, input_affine = generate('triple ellipsoid')

    # Define rotation
    angle = 0.1 * (np.pi/2)
    axis = [1,0,0]
    rotation = angle * np.array(axis)/np.linalg.norm(axis)

    # Perform rotation
    start_time = time.time()
    output_data, output_affine = rotate_reshape(input_data, input_affine, rotation) # not logical that this returns the output_affine
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    transformation = affine_matrix(rotation=rotation)
    plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)


def test_stretch(n=1, show=True):

    # Generate reference volume
    input_data, input_affine = generate('triple ellipsoid', markers=False)

    # Define transformation
    if n==1:
        stretch_factor = np.array([2.0, 2.5, 0.5])
    elif n==2:
        stretch_factor = np.array([1.0, 1.5, 1.0])
    elif n==3:
        stretch_factor = np.array([1.0, 1.1, 1.0])

    # Define output_volume
    output_shape = input_data.shape
    output_affine = input_affine

    # Perform rotation
    start_time = time.time()
    output_data = stretch(input_data, input_affine, output_shape, output_affine, stretch_factor) # specifying outputshape and affine should not be required
    end_time = time.time()

    # Display results
    if show is True:
        print('Computation time (sec): ', end_time-start_time)
        transformation = affine_matrix(pixel_spacing=stretch_factor)
        plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)

    return input_data, input_affine, output_data, output_affine, stretch_factor


def test_stretch_reshape(show=True):

    # Generate reference volume
    input_data, input_affine = generate('triple ellipsoid', markers=False)

    # Define transformation
    stretch_factor = np.array([2.0, 2.5, 0.5])

    # Perform transformation
    start_time = time.time()
    output_data, output_affine = stretch_reshape(input_data, input_affine, stretch_factor) # specifying outputshape and affine should not be required
    end_time = time.time()

    # Display results
    if show is True:
        print('Computation time (sec): ', end_time-start_time)
        transformation = affine_matrix(pixel_spacing=stretch_factor)
        plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)

    return input_data, input_affine, output_data, output_affine, stretch_factor


def test_rotate_around():

    # Generate reference volume
    input_data, input_affine = generate('ellipsoid', markers=False)

    # Define rotation
    rotation = 0.5 * np.pi/2 * np.array([1, 0, 0]) # radians
    com = center_of_mass(input_data, input_affine)

    # Define output_volume
    output_shape = input_data.shape
    output_affine = input_affine
    
    # Perform rotation
    start_time = time.time()
    parameters = np.concatenate((rotation, com))
    output_data = rotate_around(input_data, input_affine, output_shape, output_affine, parameters)
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    transformation = affine_matrix(rotation=rotation, center=com)
    plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)


def test_rotate_around_reshape():

    # Generate reference volume
    input_data, input_affine = generate('ellipsoid', markers=False)

    # Define rotation
    rotation = 0.5 * np.pi/2 * np.array([1, 0, 0]) # radians
    com = center_of_mass(input_data, input_affine)
    
    # Perform rotation
    start_time = time.time()
    output_data, output_affine = rotate_around_reshape(input_data, input_affine, rotation, com)
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    transformation = affine_matrix(rotation=rotation, center=com)
    plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)


def test_rigid(show=True):

    # Generate reference volume
    input_data, input_affine = generate('ellipsoid', markers=False)

    # Define rigid transformation
    angle = 0.5 * (np.pi/2)
    axis = [1,0,0]
    translation = np.array([0, 60, -40]) # mm
    rotation = angle * np.array(axis)/np.linalg.norm(axis)

    # Define output_volume
    output_shape = input_data.shape
    output_affine = input_affine

    # Perform rigid transformation   
    start_time = time.time()
    parameters = np.concatenate((rotation, translation))
    output_data = rigid(input_data, input_affine, output_shape, output_affine, parameters)
    end_time = time.time()

    # Display results
    if show is True:
        print('Computation time (sec): ', end_time-start_time)
        transformation = affine_matrix(rotation=rotation, translation=translation)
        plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)

    return input_data, input_affine, output_data, output_affine, parameters


def test_rigid_reshape():

    # Generate input data
    input_data, input_affine = generate('ellipsoid', markers=False)

    # Define rigid transformation
    angle = 0.5 * (np.pi/2)
    axis = [1,0,0]
    translation = np.array([0, 60, -40]) # mm
    rotation = angle * np.array(axis)/np.linalg.norm(axis)

    # Perform rigid transformation   
    start_time = time.time()
    output_data, output_affine = rigid_reshape(input_data, input_affine, rotation, translation)
    end_time = time.time()

    # Display results
    print('Computation time (sec): ', end_time-start_time)
    transformation = affine_matrix(rotation=rotation, translation=translation)
    plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)

    
def test_affine(show=True):

    # Define geometry of source data
    input_shape = np.array([300, 250, 25])   # mm
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]

    # Generate source volume data
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

    # Define affine transformation
    stretch = [1.0, 1.5, 1.5]
    translation = np.array([30, -80, -20]) # mm
    rotation_angle = 0.20 * (np.pi/2)
    rotation_axis = [0,0,1]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)

    # Define output_volume
    output_shape = input_data.shape
    output_affine = input_affine

    # Apply affine
    start_time = time.time()
    parameters = np.concatenate((rotation, translation, stretch))
    output_data = affine(input_data, input_affine, output_shape, output_affine, parameters)
    end_time = time.time()

    # Display results
    if show:
        print('Computation time (sec): ', end_time-start_time)
        transformation = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
        plot_affine_transformed(input_data, input_affine, output_data, output_affine, transformation)

    return input_data, input_affine, output_data, output_affine, parameters


def test_freeform(show=True):

    # Define geometry of source data
    input_shape = np.array([300, 250, 25])   # mm
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = 0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]

    # Generate source data
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    input_data, input_affine = generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

    # Define affine transformation
    stretch = [1.0, 1.5, 1.5]
    translation = np.array([30, -80, -20]) # mm
    rotation_angle = 0.20 * (np.pi/2)
    rotation_axis = [0,0,1]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)

    # Define output_volume (this can also be a window)
    output_shape = input_data.shape
    output_affine = input_affine

    # Apply freeform deformation
    start_time = time.time()
    #parameters = np.concatenate((rotation, translation, stretch))
    #output_data = affine(input_data, input_affine, output_shape, output_affine, parameters)
    transformation = affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)
    deformation_field_shape = (2,2,2)
    parameters = affine_to_freeform(transformation, output_shape, deformation_field_shape)
    # parameters = np.ones((2, 2, 2, 3)) # (x, y, z, d)
    # output_data = freeform(input_data, input_affine, output_shape, output_affine, parameters)
    end_time = time.time()

    # Display results
    if show:
        print('Computation time (sec): ', end_time-start_time)
        plot_freeform_transformed(input_data, input_affine, output_data, output_affine, parameters)

    return input_data, input_affine, output_data, output_affine, parameters


def test_align_translation(n=1):

    if n==1:
        input_data, input_affine, output_data, output_affine, translation = generate_translated_data_1()
    elif n==2:
        input_data, input_affine, output_data, output_affine, translation = generate_translated_data_2()
    elif n==3:
        input_data, input_affine, output_data, output_affine, translation = generate_translated_data_3()

    # Define initial values and step size
    _, _, output_pixel_spacing = affine_components(output_affine)
    initial_guess = np.array([0, 0, 0], dtype=np.float32) # mm
    gradient_step = output_pixel_spacing

    # Define optimization method
    optimization = {'method':'GD', 'options':{'gradient step': gradient_step, 'tolerance': 0.001}, 'callback':print_current}
    #optimization = {'method':'Powell', 'options':{'xtol': 1.0}, 'callback':print_current}
    #optimization = {'method':'BFGS', 'options':{'eps': gradient_step}, 'callback':print_current}

    # Define transformation
    transformation = translate

    # Align volumes
    start_time = time.time()
    try:
        translation_estimate = align(
            moving = input_data, 
            moving_affine = input_affine, 
            static = output_data, 
            static_affine = output_affine, 
            parameters = initial_guess, 
            resolutions = [4,2,1], 
            transformation = transformation,
            metric = sum_of_squares,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes. Returning initial value as current best guess..')
        translation_estimate = initial_guess
    end_time = time.time()

    # Calculate estimate of static image
    output_data_estimate = transformation(input_data, input_affine, output_data.shape, output_affine, translation_estimate)

    # Display results
    print('Ground truth parameter: ', translation)
    print('Parameter estimate: ', translation_estimate)
    print('Parameter error (%): ', 100*np.linalg.norm(translation_estimate-translation)/np.linalg.norm(translation))
    print('Computation time (mins): ', (end_time-start_time)/60.0)
    pl = plot_affine_resliced(output_data_estimate, output_affine, output_data, output_affine)
    pl.show()


def test_align_rotation(n=1):

    if n==1:
        input_data, input_affine, output_data, output_affine, rotation = test_rotate(show=False)

    # Define initial values and step size
    _, _, output_pixel_spacing = affine_components(output_affine)
    initial_guess = np.array([0, 0, 0], dtype=np.float32) # mm
    #gradient_step = np.array([np.pi/180]*3)
    gradient_step, _, _ = affine_resolution(output_data.shape, output_pixel_spacing)

    # Define optimization method
    # Define a precision for each parameter and stop iterating when the largest change for
    # any of the parameters is less than its precision. 
    # The gradient step is also the precision and the tolarance becomes unnecessary
    optimization = {'method':'GD', 'options':{'gradient step': gradient_step, 'tolerance': 0.1}, 'callback':print_current}
    #optimization = {'method':'Powell', 'options':{'xtol': 1.0}, 'callback':print_current}
    #optimization = {'method':'BFGS', 'options':{'eps': gradient_step}, 'callback':print_current}

    # Define transformation
    transformation = rotate

    # Align volumes
    start_time = time.time()
    try:
        estimate = align(
            moving = input_data, 
            moving_affine = input_affine, 
            static = output_data, 
            static_affine = output_affine, 
            parameters = initial_guess, 
            resolutions = [4,2,1], 
            transformation = transformation,
            metric = sum_of_squares,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes. Returning initial value as current best guess..')
        estimate = initial_guess
    end_time = time.time()

    # Calculate estimate of static image
    output_data_estimate = transformation(input_data, input_affine, output_data.shape, output_affine, estimate)

    # Display results
    print('Ground truth parameter: ', rotation)
    print('Parameter estimate: ', estimate)
    print('Parameter error (%): ', 100*np.linalg.norm(estimate-rotation)/np.linalg.norm(rotation))
    print('Computation time (mins): ', (end_time-start_time)/60.0)
    pl = plot_affine_resliced(output_data_estimate, output_affine, output_data, output_affine)
    pl.show()


def test_align_stretch(n=1):

    input_data, input_affine, output_data, output_affine, parameters = test_stretch(n=n, show=False)

    # Define initial values and step size
    _, _, output_pixel_spacing = affine_components(output_affine)
    initial_guess = np.array([1, 1, 1], dtype=np.float32) # mm
    _, _, step = affine_resolution(output_data.shape, output_pixel_spacing)
    tol = 0.1
    #bounds = [(tol*step[0], np.inf), (tol*step[1], np.inf), (tol*step[2], np.inf)]
    bounds = [(0.5, np.inf), (0.5, np.inf), (0.5, np.inf)]

    # Define registration method
    optimization = {'method':'GD', 'bounds':bounds, 'options':{'gradient step': step, 'tolerance': tol}, 'callback':print_current}
    transformation = stretch
    metric = sum_of_squares

    # Align volumes
    start_time = time.time()
    try:
        estimate = align(
            moving = input_data, 
            moving_affine = input_affine, 
            static = output_data, 
            static_affine = output_affine, 
            parameters = initial_guess, 
            resolutions = [4,2,1], 
            transformation = transformation,
            metric = metric,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes. Returning initial value as current best guess..')
        estimate = initial_guess
    end_time = time.time()

    # Calculate estimate of static image and cost functions
    output_data_estimate = transformation(input_data, input_affine, output_data.shape, output_affine, estimate)
    cost_after = goodness_of_alignment(estimate, transformation, metric, input_data, input_affine, output_data, output_affine, None) 
    cost_after *= 100/np.sum(np.square(output_data))
    cost_before = goodness_of_alignment(initial_guess, transformation, metric, input_data, input_affine, output_data, output_affine, None) 
    cost_before *= 100/np.sum(np.square(output_data))

    # Display results
    print('Ground truth parameter: ', parameters)
    print('Parameter estimate: ', estimate)
    print('Cost before alignment (%): ', cost_before)
    print('Cost after alignment (%): ', cost_after)
    print('Parameter error (%): ', 100*np.linalg.norm(estimate-parameters)/np.linalg.norm(parameters))
    print('Computation time (mins): ', (end_time-start_time)/60.0)
    pl = plot_affine_resliced(output_data_estimate, output_affine, output_data, output_affine)
    pl.show()


def test_align_rigid(n=1):

    if n==1:
        input_data, input_affine, output_data, output_affine, parameters = test_rigid(show=False)
    if n==2:
        input_data, input_affine, output_data, output_affine, rotation = test_rotate(show=False)
        translation = np.zeros(3, dtype=np.float32)
        parameters = np.concatenate((rotation, translation))

    # Define initial values and step size
    _, _, output_pixel_spacing = affine_components(output_affine)
    initial_guess = np.zeros(parameters.shape, dtype=np.float32) # mm
    rot_gradient_step, translation_gradient_step, _ = affine_resolution(output_data.shape, output_pixel_spacing)
    gradient_step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step))

    # Define registration
    optimization = {'method':'GD', 'options':{'gradient step': gradient_step, 'tolerance': 0.1}, 'callback':print_current}
    transformation = rigid
    metric = sum_of_squares

    # Align volumes
    start_time = time.time()
    try:
        estimate = align(
            moving = input_data, 
            moving_affine = input_affine, 
            static = output_data, 
            static_affine = output_affine, 
            parameters = initial_guess, 
            resolutions = [4,2,1], 
            transformation = transformation,
            metric = metric,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes. Returning initial value as current best guess..')
        estimate = initial_guess
    end_time = time.time()

    # Calculate estimate of static image and cost functions
    output_data_estimate = transformation(input_data, input_affine, output_data.shape, output_affine, estimate)
    cost_after = goodness_of_alignment(estimate, transformation, metric, input_data, input_affine, output_data, output_affine, None) 
    cost_after *= 100/np.sum(np.square(output_data))
    cost_before = goodness_of_alignment(initial_guess, transformation, metric, input_data, input_affine, output_data, output_affine, None) 
    cost_before *= 100/np.sum(np.square(output_data))

    # Display results
    print('Ground truth parameter: ', parameters)
    print('Parameter estimate: ', estimate)
    print('Cost before alignment (%): ', cost_before)
    print('Cost after alignment (%): ', cost_after)
    print('Parameter error (%): ', 100*np.linalg.norm(estimate-parameters)/np.linalg.norm(parameters))
    print('Computation time (mins): ', (end_time-start_time)/60.0)
    pl = plot_affine_resliced(output_data_estimate, output_affine, output_data, output_affine)
    pl.show()


def test_align_affine(n=1):

    if n==1:
        input_data, input_affine, output_data, output_affine, parameters = test_affine(show=False)

    # Define initial values and step size
    _, _, output_pixel_spacing = affine_components(output_affine)
    initial_guess = np.array([0,0,0,0,0,0,1,1,1], dtype=np.float32)
    rot_gradient_step, translation_gradient_step, stretch_gradient_step = affine_resolution(output_data.shape, output_pixel_spacing)
    step = np.concatenate((1.0*rot_gradient_step, 0.5*translation_gradient_step, stretch_gradient_step))
    bounds = [
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), 
        (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), 
        (0.5, np.inf), (0.5, np.inf), (0.5, np.inf),
    ]

    # Define registration
    optimization = {'method':'GD', 'bounds': bounds, 'options':{'gradient step': step, 'tolerance': 0.1}, 'callback':print_current}
    transformation = affine
    metric = sum_of_squares

    # Align volumes
    start_time = time.time()
    try:
        estimate = align(
            moving = input_data, 
            moving_affine = input_affine, 
            static = output_data, 
            static_affine = output_affine, 
            parameters = initial_guess, 
            resolutions = [4,2,1], 
            transformation = transformation,
            metric = metric,
            optimization = optimization,
        )
    except:
        print('Failed to align volumes. Returning initial value as current best guess..')
        estimate = initial_guess
    end_time = time.time()

    # Calculate estimate of static image and cost functions
    output_data_estimate = transformation(input_data, input_affine, output_data.shape, output_affine, estimate)
    cost_after = goodness_of_alignment(estimate, transformation, metric, input_data, input_affine, output_data, output_affine, None) 
    cost_after *= 100/np.sum(np.square(output_data))
    cost_before = goodness_of_alignment(initial_guess, transformation, metric, input_data, input_affine, output_data, output_affine, None) 
    cost_before *= 100/np.sum(np.square(output_data))

    # Display results
    print('Ground truth parameter: ', parameters)
    print('Parameter estimate: ', estimate)
    print('Cost before alignment (%): ', cost_before)
    print('Cost after alignment (%): ', cost_after)
    print('Parameter error (%): ', 100*np.linalg.norm(estimate-parameters)/np.linalg.norm(parameters))
    print('Computation time (mins): ', (end_time-start_time)/60.0)
    pl = plot_affine_resliced(output_data_estimate, output_affine, output_data, output_affine)
    pl.show()



        



if __name__ == "__main__":

    dataset=2

    # Test plotting
    # -------------
    # test_plot(1)


    # Test affine transformations
    # ---------------------------
    # test_affine_reslice(6)
    # test_affine_transform()
    # test_affine_transform_reshape()
    # test_affine_transform_and_reslice()


    # Test forward models
    # -------------------
    # test_translate(dataset)
    # test_translate_reshape()
    # test_rotate()
    # test_rotate_reshape()
    # test_stretch(n=3)
    # test_stretch_reshape()
    # test_rotate_around()
    # test_rotate_around_reshape()
    # test_rigid()
    # test_rigid_reshape()
    # test_affine()
    test_freeform()


    # Test coregistration
    # -------------------
    #test_align_translation(dataset)
    #test_align_rotation()
    # test_align_stretch(n=2)
    #test_align_rigid(n=1)
    # test_align_affine(n=1)


