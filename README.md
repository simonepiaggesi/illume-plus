# Explainable AI for Cancer Drug Response Prediction: Beyond Univariate Feature Attributions

Code and data for running ILLUME+. 

Before using the code, please [download](https://zenodo.org/records/18605674?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgxNDdhMWQ0LTFiY2ItNGNiOC04YmZiLTA2OTZkNjQyNmE4YSIsImRhdGEiOnt9LCJyYW5kb20iOiJiZWE4OGJlMjE1ZTFmOTExYzdmZWZkZWJhODAyODNhMCJ9.50gzq80ASCYnCB1-ooG7jqO-4Z65ykXq7s23Kz3telUWKF1xABfnsqowcFvP2pkNsW7Vr7pb3okktUrO3Gz4rA) the full gene expression profiles in the `data/` folder.

The code has been tested with Python 3.11.

## Reproducibility
Set up the conda environment using the provided YAML file:
   ```bash
   conda env create -f env.yml
   conda activate illume4s
   ```
To run the method, please look at [main.ipynb](main.ipynb).
