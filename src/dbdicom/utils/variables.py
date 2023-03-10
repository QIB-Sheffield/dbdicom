import math
import datetime

def str_to_seconds(dicom_tm):
    # dicom_tm is of the form 'HHMMSS.FFFFFF'
    # Split the seconds into seconds and fractional seconds
    seconds, fractional_seconds = dicom_tm.split('.')
    # Convert the hours, minutes, and seconds to integers
    hours = int(seconds[:2])
    minutes = int(seconds[2:4])
    seconds = int(seconds[4:])
    # Convert the fractional seconds to a decimal
    fractional_seconds = float('.' + fractional_seconds)
    # Create a datetime object representing the time
    seconds_since_midnight = (hours * 3600) + (minutes * 60) + seconds + fractional_seconds
    return seconds_since_midnight

def seconds_to_str(seconds_since_midnight):
    hours = math.floor(seconds_since_midnight/3600)
    minutes = math.floor((seconds_since_midnight-hours*3600)/60)
    seconds = math.floor(seconds_since_midnight-hours*3600-minutes*60)
    fractional_seconds = round(seconds_since_midnight-hours*3600-minutes*60-seconds, 6)
    hours = str(hours)
    minutes = str(minutes)
    seconds = str(seconds)
    fractional_seconds = str(fractional_seconds)
    _, fractional_seconds = fractional_seconds.split('.')
    fractional_seconds = fractional_seconds.ljust(6,'0')
    return hours + minutes + seconds + '.' + fractional_seconds

def time_to_seconds(tm):
    hours = tm.hour
    minutes = tm.minute
    seconds = tm.second
    fractional_seconds = tm.microsecond / 1000000.0
    seconds_since_midnight = (hours * 3600) + (minutes * 60) + seconds + fractional_seconds
    return seconds_since_midnight

def seconds_to_time(seconds_since_midnight):
    hours = math.floor(seconds_since_midnight/3600)
    minutes = math.floor((seconds_since_midnight-hours*3600)/60)
    seconds = math.floor(seconds_since_midnight-hours*3600-minutes*60)
    fractional_seconds = round(seconds_since_midnight-hours*3600-minutes*60-seconds, 6)
    microseconds = fractional_seconds*1000000.0
    return datetime.time(int(hours), int(minutes), int(seconds), int(microseconds))

def time_to_str(tm):
    hours = tm.hour
    minutes = tm.minute
    seconds = tm.second
    fractional_seconds = tm.microsecond / 1000000.0   
    hours = str(hours)
    minutes = str(minutes)
    seconds = str(seconds)
    fractional_seconds = str(fractional_seconds)
    _, fractional_seconds = fractional_seconds.split('.')
    fractional_seconds = fractional_seconds.ljust(6,'0')
    return hours + minutes + seconds + '.' + fractional_seconds 

def date_to_str(tm):
    year = str(tm.year).rjust(4, '0')
    month = str(tm.month).rjust(2, '0')
    day = str(tm.day).rjust(2, '0')
    return year + month + day

def datetime_to_str(dt):
    date = date_to_str(dt.date())
    time = time_to_str(dt.time())
    return date + time



if __name__ == "__main__":

    sec = 13*60*60 + 12*60 + 40 + 0.03
    dcm = '131240.030000'
    tim = datetime.time(13, 12, 40, 30000)
    date = datetime.date(2023, 3, 1)
    date_str = '20230301'
    dt = datetime.datetime(2023, 3, 1, 13, 12, 40, 30000)
    dt_str = '20230301131240.030000'

    assert str_to_seconds(dcm) == sec
    assert seconds_to_str(sec) == dcm
    assert str_to_seconds(seconds_to_str(sec)) == sec
    assert seconds_to_str(str_to_seconds(dcm)) == dcm
    assert time_to_seconds(tim) == sec
    assert seconds_to_time(sec) == tim
    assert time_to_seconds(seconds_to_time(sec)) == sec
    assert seconds_to_time(time_to_seconds(tim)) == tim
    assert date_to_str(date) == date_str
    assert time_to_str(tim) == dcm
    assert datetime_to_str(dt) == dt_str

    print('Module passed all tests')
