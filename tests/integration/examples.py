import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from dbdicom.register import DbRegister


test_folder = os.path.join(os.path.dirname(__file__), 'data')
test_export_folder = os.path.join(os.path.dirname(__file__), 'results')




def test_set_array_invert():

    folder = DbRegister(test_folder2).open() # enhancement: open automatically on init
    series = folder.series(12) # T1-MOLLI multi-TI
    array, _ = series.array()
    invert = series.copy().set_array(-array)
    array_invert, _ = invert.array()
    print('Round-off error (%): ', 100*np.sum(np.square(array + array_invert))/np.sum(np.square(array)))
    col = 3
    fig = plt.figure(figsize=(16,16))
    i=0
    fig.add_subplot(1,col,i+1)
    plt.imshow(array[0,:,:])
    plt.colorbar()
    i=1
    fig.add_subplot(1,col,i+1)
    plt.imshow(array_invert[0,:,:])
    plt.colorbar()
    i=2
    fig.add_subplot(1,col,i+1)
    plt.imshow(array[0,:,:]+array_invert[0,:,:])
    plt.colorbar()
    plt.show()

    folder.restore()

def test_sort_series():

    folder = DbRegister(test_folder2).open()
    series = folder.series(12) # T1-MOLLI multi-TI
#    data = series.sort(['SliceLocation','InversionTime', 'PatientName'])
    data = series.sort(['SliceLocation','InversionTime', 'PatientName'])
    print(data[0,0,0,0].__class__.__name__)
    print(data.shape)
    print(data[0,0,0,0].SliceLocation)
    print(data[0,0,0,0].InversionTime)
    print(data[0,0,0,0].PatientName)
    loc = [data[z,0,0,0].SliceLocation for z in range(data.shape[0])]
    print(loc)

def test_read_series_array():

    folder = DbRegister(test_folder2).open()
    series = folder.series(12) # T1-MOLLI multi-TI
    array, data = series.array(['SliceLocation','InversionTime', 'PatientName'], pixels_first=True)
    print(array.shape)
    print(data.shape)
    print([data[0,0,0,0].SliceLocation, data[0,0,0,0].InversionTime, data[0,0,0,0].PatientName])


# test_read_series_array()
# test_sort_series()

# test_set_array_invert()
# test_set_array_mip()