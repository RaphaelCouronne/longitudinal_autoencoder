# longitudinal_autoencoder
Code for the paper Longitudinal self-supervision to disentangleinter-patient variability from disease progression, based on PyTorch 1.7.

Requirements : 
- environment.yml
- fast-soft-sort (https://github.com/google-research/fast-soft-sort)
- leaspy (https://gitlab.com/icm-institute/aramislab/leaspy/)

Datasets : 
- Starmen (http://doi.org/10.5281/zenodo.5081988)
- ADNI cognitive scores http://adni.loni.usc.edu/
- ADNI MRIs http://adni.loni.usc.edu/

Run a test on scalar data :
- run_test.sh : specify a .csv with first column as ID, second column TIME (e.g. Age), and other columns as monotonic clinical markers.

Launch : 
- sh experiments/run_figure2_table.sh
- python experiments/run_figure2_table_analysis.py
- sh experiments/run_figure3_adnicognitive.sh
- python experiments/run_figure3_adnicognitive_analysis.py


