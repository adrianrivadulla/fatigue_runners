# Fatigue study

[![DOI](https://zenodo.org/badge/302002826.svg)](https://doi.org/10.5281/zenodo.5532714)

The scripts within this repo can replicate the analysis conducted for 

Rivadulla, A. R., Sheehy, Z., Chen, X., Cazzola, D., Trewartha, G., & Preatoni, E. (2025). Does preferred technique influece how kinematics change during a run to exhaustion? - A cluster based approach. under review.

## Getting Started

These instructions will get a copy of the project up and running on your local machine for development and testing purposes.

### Pre-requisites

- Python >= 3.8

### Installing

- Clone this repository and enter it:

```Shell
   git clone https://github.com/adrianrivadulla/fatigue_runners.git
```

or download and unzip this repository.

- Set up the environment. Using Anaconda is recommended:

Navigate to the fatigue_runners directory and create a version environment with the environment.yml file provided:

 ```Shell
     cd /path/to/fatigue_runners
     conda env create -f environment.yml
 ```

### Getting the data

Download the dataset from https://researchdata.bath.ac.uk/1550/. Create a `data` directory by unzipping the dataset within the project root. Please refer to the dataset description in the link for further details on how the data and files are structured.


### Usage


- Activate the environment:

```Shell
    conda activate fatigue_runners_env
```

- Run fatigue_analysis.py:

```Shell
    python fatigue_analysis.py
```

The script should replicate the figures presented in the paper which will get stored in the report dir alongside some additional figures.
There might be some seaborn, matplotlib or pandas warnings, but they can be ignored as they should not prevent the script from running and they should not affect the results.

Documentation for the different scripts and functions has been generated using Copilot and it can be brief in some cases but hopefully enough to understand what is happening. 

## License

Copyright (c) 2025 Adrian R Rivadulla

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program (see gpl.txt and lgpl.txt). If not, see <https://www.gnu.org/licenses/>.

# Contact
For questions about our paper or code, please contact [Adrian R](mailto:arr43@bath.ac.uk).
