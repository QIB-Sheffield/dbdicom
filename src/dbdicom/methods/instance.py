import numpy as np
import nibabel as nib

from dbdicom.record import DbRecord


class Instance(DbRecord):

    def get_pixel_array(self):

        ds = self.get_dataset()
        return ds.get_pixel_array()

    def set_pixel_array(self, array):

        ds = self.get_dataset()
        ds.set_pixel_array(array)
        self.set_dataset(ds)

    def map_mask_onto(self, target):
        return map_mask_onto(self, target)

    def export_as_csv(*args, **kwargs):
        export_as_csv(*args, **kwargs)


def map_mask_onto(record, target):
    """Map non-zero image pixels onto a target image.
    
    Overwrite pixel values in the target"""

    dsr = record.get_dataset()
    dst = target.get_dataset()

    # Create a coordinate array of non-zero pixels
    coords = np.transpose(np.where(dsr.get_pixel_array() != 0)) 
    coords = [[coord[0], coord[1], 0] for coord in coords] 
    coords = np.array(coords)

    # Determine coordinate transformation matrix
    affineSource = dsr.affine_matrix()
    affineTarget = dst.affine_matrix()
    sourceToTarget = np.linalg.inv(affineTarget).dot(affineSource)

    # Apply coordinate transformation
    coords = nib.affines.apply_affine(sourceToTarget, coords)
    coords = np.round(coords, 3).astype(int)
    x = tuple([coord[0] for coord in coords if coord[2] == 0])
    y = tuple([coord[1] for coord in coords if coord[2] == 0])

    # Set values in the target image
    # Note - replace by actual values rather than 1 & 0.
    array = np.zeros((record.Rows, record.Columns))
    array[(x, y)] = 1.0
    result = record.new_sibling()
    result.set_pixel_array(array)

    return result


def export_as_csv(record, directory=None, filename=None, columnHeaders=None):
    """Export 2D pixel Array in csv format"""

    if directory is None: 
        directory = record.directory(message='Please select a folder for the csv data')
    if filename is None:
        filename = record.SeriesDescription
    filename = os.path.join(directory, filename + '.csv')
    table = record.array()
    if columnHeaders is None:
        columnHeaders = []
        counter = 0
        for _ in table:
            counter += 1
            columnHeaders.append("Column" + str(counter))
    df = pd.DataFrame(np.transpose(table), columns=columnHeaders)
    df.to_csv(filename, index=False)
