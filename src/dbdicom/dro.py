# Importing annotations to handle or sign in import type hints
from __future__ import annotations

import numpy as np
import dbdicom as db
from dbdicom.types.database import Database
from dbdicom.types.series import Series
from dbdicom.utils import image


def T1_mapping_vFATR(spacing = (15, 15, 20), fov = (300, 250, 120), T1min = 600, S0min = 100, vFA = [2.0, 20.0], vTR = [5.0,15.0])->Series:
    """Synthetic T1-mapping data with variable TR and FA

    Args:
        spacing (tuple, optional): x, y, z pixel spacing in mm. Defaults to (1.5, 1.5, 2.0).
        fov (tuple, optional): x, y, z field of view in mm. Defaults to (300, 250, 120).
        T1min (int, optional): smallest T1 in msec. Defaults to 600.
        S0min (int, optional): smallest S0 in a.u. Defaults to 100.
        vFA (list, optional): variable flip angle values in degrees. Defaults to [2.0, 5.0, 10.0, 15.0, 20.0].
        vTR (list, optional): variable repetition time values in msec. Defaults to [2.0, 3.0, 4.0, 5.0, 10.0, 15.0].

    Returns:
        dbdicom.Series: A series with appropriate array and header data.
    """
    ellipsoid = image.ellipsoid(fov[0]/2, fov[1]/2, fov[2]/2, spacing=spacing, levelset=True)
    ellipsoid = 1 + ellipsoid - np.amin(ellipsoid)
    T1 = T1min*ellipsoid
    S0 = S0min*ellipsoid
    array = np.empty((T1.shape[0], T1.shape[1], T1.shape[2], len(vFA), len(vTR)))
    for j, TR in enumerate(vTR):
        Ej = np.exp(-TR/T1)
        for i, FA in enumerate(vFA):
            ci = np.cos(FA*np.pi/180)
            array[:,:,:,i,j] = S0 * (1-Ej) / (1-ci*Ej)
    
    coords = {
        'SliceLocation': spacing[2]*np.arange(array.shape[2]),
        'FlipAngle': np.array(vFA),
        'RepetitionTime': np.array(vTR),
    }
    v0, v1 = np.amin(array), np.amax(array)
    series = db.as_series(array, gridcoords=coords, PixelSpacing=[spacing[1],spacing[0]], WindowWidth=v1-v0, WindowCenter=(v0+v1)/2)
    series.patient().PatientName = 'Ellipsoid'
    series.study().StudyDescription = 'Synthetic'
    series.SeriesDescription = 'T1 mapping variable TR and FA'

    return series
    

def ellipsoid(a, b, c, spacing=(1., 1., 1.), levelset=False)->Series:
    """
    Generates ellipsoid with semimajor axes aligned with grid dimensions
    on grid with specified `spacing`.

    Args:
        a (float): Length of semimajor axis aligned with x-axis.
        b (float): Length of semimajor axis aligned with y-axis.
        c (float): Length of semimajor axis aligned with z-axis.
        spacing (tuple of floats, length 3): Spacing in (x, y, z) spatial dimensions. Defaults to (1,1,1)
        levelset (bool): If True, returns the level set for this ellipsoid (signed level set about zero, with positive denoting interior) as np.float64. False returns a binarized version of said level set. Defaults to False.
    
    Returns:
        dbdicom.Series: A series with appropriate array and header data.
    
    Note:
        The interface and the array generation is taken directly from skimage but the core function is copied into dbdicom utilities to avoid bringing in skimage as an essential dependency.
    """
    arr = image.ellipsoid(a, b, c, spacing=spacing, levelset=levelset)
    coords = {'SliceLocation': spacing[2]*np.arange(arr.shape[2])}
    v0, v1 = np.amin(arr), np.amax(arr)
    series = db.as_series(arr, gridcoords=coords, PixelSpacing=[spacing[1],spacing[0]], WindowWidth=v1-v0, WindowCenter=(v0+v1)/2)
    affine = np.array(
        [[spacing[1], 0., 0., 0.],
         [0., spacing[0], 0., 0.],
         [0., 0., spacing[2], 0.],
         [0., 0., 0., 1.]]
    )
    series.set_affine(affine)
    series.patient().PatientName = 'Ellipsoid'
    series.study().StudyDescription = 'Synthetic'
    series.SeriesDescription = 'Levelset ellipsoid'
    return series


def double_ellipsoid(a, b, c, spacing=(1., 1., 1.), levelset=False)->Series:
    """
    Generates a double ellipsoid with semimajor axes aligned with grid dimensions
    on grid with specified `spacing`.

    Args:
        a (float): Length of semimajor axis aligned with x-axis.
        b (float): Length of semimajor axis aligned with y-axis.
        c (float): Length of semimajor axis aligned with z-axis.
        spacing (tuple of floats, length 3): Spacing in (x, y, z) spatial dimensions. Defaults to (1,1,1)
        levelset (bool): If True, returns the level set for this ellipsoid (signed level set about zero, with positive denoting interior) as np.float64. False returns a binarized version of said level set. Defaults to False.

    Returns:
        dbdicom.Series: A series with appropriate array and header data.
    
    Note:
        The interface and the array generation is taken directly from skimage, but the core function is copied into dbdicom utilities to avoid bringing in skimage as an essential dependency.
    """
    arr = image.ellipsoid(a, b, c, spacing=spacing, levelset=levelset)
    coords = {'SliceLocation': spacing[2]*np.arange(arr.shape[2])}
    arr = np.concatenate((arr[:-1, ...], arr[2:, ...]), axis=0)
    v0, v1 = np.amin(arr), np.amax(arr)
    series = db.as_series(arr, gridcoords=coords, PixelSpacing=[spacing[1],spacing[0]], WindowWidth=v1-v0, WindowCenter=(v0+v1)/2)
    series.patient().PatientName = 'Ellipsoid'
    series.study().StudyDescription = 'Synthetic'
    series.SeriesDescription = 'Levelset ellipsoid'
    return series


def database_hollywood()->Database:
    """Create an empty toy database for demonstration purposes.

    Returns:
        Database: Database with two patients, two studies per patient and two empty series per study.

    See Also:
        :func:`~database`

    Example:
        >>> database = db.dro.database_hollywood()
        >>> database.print()
        ---------- DATABASE --------------
        Location:  In memory
            Patient James Bond
                Study MRI [19821201]
                    Series 001 [Localizer]
                        Nr of instances: 0
                    Series 002 [T2w]
                        Nr of instances: 0
                Study Xray [19821205]
                    Series 001 [Chest]
                        Nr of instances: 0
                    Series 002 [Head]
                        Nr of instances: 0
            Patient Scarface
                Study MRI [19850105]
                    Series 001 [Localizer]
                        Nr of instances: 0
                    Series 002 [T2w]
                        Nr of instances: 0
                Study Xray [19850106]
                    Series 001 [Chest]
                        Nr of instances: 0
                    Series 002 [Head]
                        Nr of instances: 0
        ---------------------------------
    """
    hollywood = db.database()
    hollywood.mute()

    james_bond = hollywood.new_patient(PatientName='James Bond')
    james_bond_mri = james_bond.new_study(StudyDescription='MRI', StudyDate='19821201')
    james_bond_mri_localizer = james_bond_mri.new_series(SeriesDescription='Localizer')
    james_bond_mri_T2w = james_bond_mri.new_series(SeriesDescription='T2w')
    james_bond_xray = james_bond.new_study(StudyDescription='Xray', StudyDate='19821205')
    james_bond_xray_chest = james_bond_xray.new_series(SeriesDescription='Chest')
    james_bond_xray_head = james_bond_xray.new_series(SeriesDescription='Head')

    scarface = hollywood.new_patient(PatientName='Scarface')
    scarface_mri = scarface.new_study(StudyDescription='MRI', StudyDate='19850105')
    scarface_mri_localizer = scarface_mri.new_series(SeriesDescription='Localizer')
    scarface_mri_T2w = scarface_mri.new_series(SeriesDescription='T2w')
    scarface_xray = scarface.new_study(StudyDescription='Xray', StudyDate='19850106')
    scarface_xray_chest = scarface_xray.new_series(SeriesDescription='Chest')
    scarface_xray_head = scarface_xray.new_series(SeriesDescription='Head')

    return hollywood

if __name__ == '__main__':
    T1_mapping_vFATR()
