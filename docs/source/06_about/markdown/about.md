# About ***dbdicom***

## Why DICOM?

``*[...] after 2 hours of reading, I still cannot figure out how to determine the 3D orientation of a multi-slice (Supplement 49) DICOM file. I'm sure it is in there somewhere, but if this minor factoid can't be deciphered in 2 hours, then the format and its documentation is too intricate.*''. Robert W. Cox, PhD, Director, Scientific and Statistical Computing Core, National Institute of Mental Health [link](https://afni.nimh.nih.gov/pub/dist/doc/nifti/nifti_revised.html).

This echoes a common frustration for anyone who has ever had a closer to look at DICOM. DICOM seems to make simple things very difficult, and the language often feels outdated to modern data scientists. 

But there are good reasons for that. DICOM not only retains imaging data, but also all other relevant data about the subject and context in which the data are taken. Detailing provenance of the data and linkage to other data is critical in radiology, but the nature of these meta data is very broad, complex and constantly changing. Storing them in some consistent and standardised way that is future proof therefore requires a systematic approach and some necessary level of abstraction. 

DICOM does this well and has for that reason grown to be the single accepted standard in medical imaging. This also explains the outdated look and feel. DICOM standardises not only the format, but also the language of medical imaging. And successful standards, by definition, don't change.

## Why ***dbdicom***?

Reading and especially writing DICOM data remains a challenging enterprise for the practicing data scientist. A typical image processing pipeline might use the excellent python package `pydicom` for extracting image arrays and any required header information from DICOM data, but will then write out the results in more manageable format such as nifty. In the process the majority of header information will have to be discarded, including detailed imaging parameters and linkage between original and derived images, follow-up studies, etc.

The practice of converting outputs in a lossy image format may be sufficient in the early stages of method development, but forms a major barrier to research or deployment of these processing methods in a real-world context. This requires results in DICOM format so they can be linked to other data of the same patients, integrated in the radiological workflow, and reviewed and edited through integrated radiological viewers. Integration of datasets ensures that all derived data are properly traceable to the source, and can be compared between subjects and within a subject over time. It also allows to test for instance whether a new (expensive) imaging method provides an *additive* benefit over and above (cheap) data from medical history, clinical exams or blood tests. 

DICOM integration of processing outputs is typically performed by DICOM specialists in the private sector, for new products that have proven clinical utility. However, this requires a major separate investment, delays the point of real-world validation until after commercialisation and massively increases the risk of costly late-stage failures. 


## What is ***dbdicom***?

`dbdicom` is a programming interface that makes reading and writing DICOM data intuitive for the practicing medical imaging scientist working in Python. DICOM-native language and terminology is hidden and replaced by concepts that are more natural for those developing in Python. The documentation therefore does not reference confusing DICOM concepts such as composite information object definitions, application entities, service-object pairs, unique identifiers, etc.

`dbdicom` wraps around DICOM using a language and code structure that is native to the 2020's. This should allow DICOM integration from the very beginning of development of new image processing methods, which means they can be deployed in clinical workflows from the very beginning. It also means that any result you generate can easily be integrated in open access DICOM databases and can be visualised along with any other images of the same subject with a standard DICOM viewer such as [OHIF](https://ohif.org/).

`dbdicom` is developed by through the [UKRIN-MAPS](https://www.nottingham.ac.uk/research/groups/spmic/research/uk-renal-imaging-network/ukrin-maps.aspx) project of the UK renal imaging network, which aims to provide clinical translation of quantitative renal MRI on a multi-vendor platform. UKRIN-MAPS is funded by the UK's [Medical Research Council](https://gtr.ukri.org/projects?ref=MR%2FR02264X%2F1).

## Acknowledgements

`dbdicom` relies heavily on `pydicom` for read/write of individual DICOM files, with some additional features provided by `nibabel` and `dcm4che`. Basic array manipulation is provided by `numpy`, and sorting and tabulating of data by `pandas`. Export to other formats is provided by `matplotlib`.