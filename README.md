This repository contains scripts used for fitting a parametrized model of gravitationally self-regulated disks to multi-wavelength dust-continuum images. See descriptions of this model in Xu (2022).

If you use scripts and data in this repository, please cite Xu (2022, https://arxiv.org/abs/2203.00941).

## Example
See example.ipynb.

If you want to use this model on observations with different image formats, you may need to slightly modify the DiskImage class in disk_model.py.

## Data
This repository does not contain the data for fitted systems in Xu (2022) due to github's file size limit. However, these data can be downloaded by running ./stats_and_visualizations/download_fitted_systems.py. Alternatively, you can manually download the data from https://doi.org/10.7910/DVN/JDZAGY and store them in ./data/fitted_systems. For examples of using these data, see notebooks in ./stats_and_visualizations.
