import numpy as np
import dbdicom as db
from dbdicom.wrappers import numpy


# This function would become part of the dbdicom package
# Would need some generalization so it also works with multiple arguments, more general records, etc
def run_on_digital_ocean(token, droplet, dbfunction, local_series):

    # Export data to remote location
    path = 'somehow/derived/from/token/and/droplet'
    local_series.export_as_dicom(path)

    # Create a new database at the remote location
    remote_database = db.database(path)
    remote_series = remote_database.series()[0]

    # Perform computation on remote series
    mip_remote = dbfunction(remote_series)

    # Import results back in the local database
    local_database = local_series.database()
    local_database.import_dicom(mip_remote.files())
    local_series = local_database.series()[0]

    # Return results as dbdicom series
    return local_series


def test_run_on_digital_ocean():

    # Open a local database
    path = 'path/to/dicom/data'
    database = db.database(path)
    series = database.series()[0]

    # Define remote location
    token = 'secret_key'
    droplet = 'datalocation'

    # Perform the same calculation remotely and locally
    mip_remote = run_on_digital_ocean(token, droplet, numpy.maximum_intensity_projection, series)
    mip_local = numpy.maximum_intensity_projection(series)

    # Check if results are the same
    mip_remote_array = mip_remote.array()
    mip_local_array = mip_local.array()

    assert np.array_equal(mip_remote_array, mip_local_array)

