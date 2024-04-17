"""
Some utilities for writing automated pipelines.
"""



def input_series(database, series_desc, study_desc=None):
    """Select a list of series for processing, and a study for saving the results"""

    # Make sure the input is a list for convenience
    lst = True
    if not isinstance(series_desc, list):
        lst = False
        series_desc = [series_desc]

    # Find series and check if valid
    input_series = []
    for desc in series_desc:
        database.message('Finding input series ' + desc)
        series = database.series(SeriesDescription=desc)
        if series == []:
            return None, None
        elif len(series) > 1:
            msg = 'Multiple series found with the description: ' + desc + '\n'
            msg += 'Please rename the others so there is only one.'
            database.dialog.information(msg)
            return None, None
        else:
            series = series[0]
        input_series.append(series)

    if study_desc is None:
        # If the input was a list, return a list - else return a scalar.
        if lst:
            return input_series
        else:
            return input_series[0]
        
    # Find study and check if valid.
    database.message('Finding export study ' + study_desc)
    studies = database.studies(StudyDescription=study_desc)
    if studies == []:
        study = input_series[0].new_pibling(StudyDescription=study_desc)
    elif len(studies) > 1:
        msg = 'Multiple studies found with the same description: ' + study_desc + '\n'
        msg += 'Please rename the others so there is only one, or choose another study for the output.'
        database.dialog.information(msg)
        return None, None
    else:
        study = studies[0]

    # If the input was a list, return a list - else return a scalar.
    if lst:
        return input_series, study
    else:
        return input_series[0], study