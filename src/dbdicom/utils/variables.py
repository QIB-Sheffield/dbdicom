from datetime import datetime

# Currently not in use - consider converting TM values
# automatically
def dicom_tm_to_datetime(dicom_tm):
    # Split the string into hours, minutes, and seconds
    hours, minutes, seconds = dicom_tm.split(':')
    # Split the seconds into seconds and fractional seconds
    seconds, fractional_seconds = seconds.split('.')
    # Convert the hours, minutes, and seconds to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    # Convert the fractional seconds to a decimal
    fractional_seconds = float('.' + fractional_seconds)
    # Create a datetime object representing the time
    tm = datetime(1900, 1, 1, hours, minutes, seconds, int(fractional_seconds * 1000000))
    return tm


def datetime_to_dicom_tm(tm):
    # Get the hours, minutes, and seconds as integers
    hours = tm.hour
    minutes = tm.minute
    seconds = tm.second
    # Get the fractional seconds as a decimal
    fractional_seconds = tm.microsecond / 1000000.0
    # Format the time as a DICOM TM string
    dicom_tm = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fractional_seconds:06f}"
    return dicom_tm
