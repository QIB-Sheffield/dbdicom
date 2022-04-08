from setuptools import setup
import urllib.request
import json

# For more information about uploading the python package to PyPI, please check the link:
# https://github.com/judy2k/publishing_python_packages_talk

# Use README.md as the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
    print(long_description)

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

# Get latest version published online in PYPI (https://pypi.org/project/dbdicom/) 
# and increment 0.0.1 (or other) so that it's uploaded correctly during Github Actions
contents = urllib.request.urlopen('https://pypi.org/pypi/dbdicom/json').read()
data = json.loads(contents)
LATEST_VERSION = data['info']['version']
latest_major, latest_minor, latest_patch = LATEST_VERSION.split(".")
new_major = "0"
new_minor = "0"
new_patch = str(int(latest_patch) + 1)  # The authors can modify this to be minor or major versions instead

NEW_VERSION = new_major + "." + new_minor + "." + new_patch

if __name__ == '__main__':
    setup(
        name="dbdicom",
        version=NEW_VERSION,
        author="Joao Almeida e Sousa and Steven Sourbron",
        author_email="j.g.sousa@sheffield.ac.uk, s.sourbron@sheffield.ac.uk",
        description="Reading and writing DICOM databases",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/QIB-Sheffield/dbdicom",
        license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
        python_requires='>=3.6, <4',
        packages=['dbdicom', 'dbdicom.classes'],
        install_requires=required,
        include_package_data=True,
        keywords=['python', "medical imaging", "DICOM"],
        # Classifiers - the purpose is to create a wheel and upload it to PYPI
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Environment :: Console',
            'Operating System :: OS Independent',

            'Programming Language :: Python :: 3',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate you support Python 3. These classifiers are *not*
            # checked by 'pip install'. See instead 'python_requires' below.
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',

            # Pick your license as you wish
            'License :: OSI Approved :: Apache Software License',
        ],
    )
