# GeoFabrics

The GeoFabrics and associated sub-packages include routines and classes for combining point (i.e. LiDAR), vector (i.e. catchment of interest, infrastructure) and raster (i.e. reference DEM) to generate a hydrologically conditioned raster. 

GeoFabrics also contains support for downloading all LiDAR tiles within a spectified polygon (catchment region) from OpenTopography. This functionality is contained in the lidar_fetch module, and can be used independently of the other geofabrics modules. An example of its usage can be found in tests/test_lidar_fetch.

## import into a conda environment
You can use this package by using pip to install the package and dependencies using the following commands (say in a conda terminal to add it to that environment), where the environment.yml is from the root of this repository.

```python
conda env create -f environment.yml
conda activate geofabrics
pip install git+https://github.com/rosepearson/GeoFabrics
```

## Basic instructions for use
The GeoFabricsGenerator class is the main entry point. This can either be accessed by importing the package then using the class directly, or through the main.py script in the root src directory. 

### importing GeoFabrics
Basic code stub looks like:
```python
from geofabrics import processor
import json
with open(r'path\to\file\\', 'r') as file_pointer:
            instructions = json.load(file_pointer)
runner = processor.GeoFabricsGenerator(instructions)
runner.run()
```
### main.py script
In the conda environment defined in the root\environment.yml, run the following:

`python src\main.py --instructions full\path\to\instruction.json`

## Tests
Tests exist for stand alone functionality (i.e. fetch lidar), and complete processing chain (i.e. creating a DEM from LiDAR files within a shapefile). A 'benchmark_dem.nc' is uploaded for each test when a DEM is generated. This is stored using git LTS as these file are not human readable. 

### Automated testing
[Github Actions](https://docs.github.com/en/actions) are used to run tests after each push to remote (i.e. github). [Miniconda](https://github.com/marketplace/actions/setup-miniconda) from the GitHub Actions marketplace is used to install the package dependencies. Linting with [Flake8](https://github.com/py-actions/flake8) and testing with [PyTest](https://docs.pytest.org/en/6.2.x/contents.html) is then performed. 

Check the actions tab after you push to check if your tests run successfully.

### Running tests locally
In the conda environment defined in the root\environment_[windows|linux].yml, run the following in the repository root folder:

1. to run individual tests
`python -m tests.test1.test1` or `python -m tests.test_lidar_fetch.test_lidar_fetch`

2. to run all tests
`python -m unittest`

## Instructions for use in spyder
If you are using spyder you can make the following changes to run main and test1.py using Run (F5)

#### Running main

Go to 'Run>Configuration per file...' and check the **Command line options** under **General settings**. Enter the following:

`--instructions full\path\to\your\instruction.json`

See image below: 

![image](https://user-images.githubusercontent.com/22883860/123566757-97a43a00-d814-11eb-9e3e-1d2468145e3d.png)

### Running test1

Go to 'Run>Configuration per file...' and check the **The following directory** under **Working directory settings** and specify the root of the repository with no quotes.

`full\path\to\the\repository\root`

See image below: 

![image](https://user-images.githubusercontent.com/22883860/123900473-3ff50280-d9bd-11eb-8123-e8b6e28d46b2.png)


## Related material
Scripts are included in separate repos. See https://github.com/rosepearson/Hydrologic-DEMs-scripts for an example.
