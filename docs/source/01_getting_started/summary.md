# Summary

The DICOM format is the universally recognised standard for medical imaging, but reading and writing DICOM data remains a challenging task for most data scientists. 

The excellent python package `pydicom` is very commonly used and well-supported, but it is limited to reading and writing individual files, and still requires a fairly high level of understanding of DICOM to ensure compliance with the standard. 

`dbdicom` wraps around `pydicom` to provide an intuitive programming interface for reading and writing data from entire DICOM databases, replacing unfamiliar DICOM-native concepts by language and notations that will be more familiar to data scientists. 

The sections below list some basic uses of `dbdicom`. The package is currently deployed in several larger scale multicentre clinical studies led by the authors, such as the [iBEAt study](https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-020-01901-x) and the [AFiRM study](https://www.uhdb.nhs.uk/afirm-study/). The package will continue to be shaped through use in these studies and we expect it will attain a more final form when these analysis pipelines are fully operational.