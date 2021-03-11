from setuptools import setup

VERSION = '0.1.1' 
DESCRIPTION = 'AGAGE 12-box model'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Setting up
setup(
        name="py12box", 
        version=VERSION,
        author="Matt Rigby",
        author_email="matt.rigby@bristol.ac.uk",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        packages=["py12box"],
        package_dir={'py12box': 'py12box'},
        include_package_data=True,
        url="https://github.com/mrghg/py12box",
        install_requires=requirements,        
        keywords=['python', 'atmosphere', 'greenhouse gas'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Natural Language :: English",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Unix",
        ],
        python_requires=">=3.7"
)