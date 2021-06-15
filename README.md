# cneuromod_alpha2_ISC

Inter-subject correlation (ISC) of movie10 data from the cneuromod2020-alpha2 release.

For an introduction to inter-subject correlation, please see
[Nastase et al. (2019), _SCAN_](https://academic.oup.com/scan/article/14/6/667/5489905).

This project duplicates some code from the [BrainIAK](https://brainiak.org/) project.
We encourage users interested in applying inter-subject correlation more broadly to check out
[their tutorials](https://brainiak.org/tutorials/).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Where the dataset will be installed
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project adapted fromthe <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
