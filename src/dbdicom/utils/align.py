import numpy as np
import scipy.optimize as opt
import scipy.ndimage as ndi


# default metrics
# ---------------

def least_squares(static, transformed):
    lsnorm = np.linalg.norm(static, transformed)
    return lsnorm**2



# default transformations
# -----------------------

def translate(moving, vector):
    return ndi.shift(moving, vector)



# Generic objective function
def _objective_function(
        params, 
        moving, 
        static, 
        transformation, 
        metric, 
        moving_mask, 
        static_mask, 
        transformation_args, 
        metric_args):

    # Transform the moving image
    if transformation_args is None:
        transformed = transformation(moving, params)
    else:
        transformed = transformation(moving, params, transformation_args)

    # If a moving mask is provided, transform this as well
    if moving_mask is not None:
        if transformation_args is None:
            transformed_mask = transformation(moving_mask, params)
        else:
            transformed_mask = transformation(moving_mask, params, transformation_args)
        transformed_mask[transformed_mask > 0.5] = 1
        transformed_mask[transformed_mask <= 0.5] = 0

    # Get the indices where the mask is non-zero
    ind = None
    if static_mask is not None and moving_mask is not None:
        ind = np.where(static_mask==1 and moving_mask==1)
    elif static_mask is not None:
        ind = np.where(static_mask==1) 
    elif moving_mask is not None:
        ind = np.where(moving_mask==1)
        
    # Calculate the cost function         
    if ind is None:
        if metric_args is None:
            return metric(static, transformed)
        else:
            return metric(static, transformed, metric_args)
    else:
        if metric_args is None:
            return metric(static[ind], transformed[ind])
        else:
            return metric(static[ind], transformed[ind], metric_args)

# Generalize for volumes with different geometries
# include affine arguments and overlay moving (+mask) on static before minimizing
def align(
        moving, 
        static, 
        params = np.zeros(3), 
        transformation = translate, 
        metric = least_squares, 
        moving_mask = None, 
        static_mask = None, 
        transformation_args = None, 
        metric_args = None):
    
    args = (
        moving, 
        static, 
        transformation, 
        metric, 
        moving_mask, 
        static_mask, 
        transformation_args, 
        metric_args)
    
    res = opt.minimize(_objective_function, params, args=args)


if __name__ == "__main__":
    pass

