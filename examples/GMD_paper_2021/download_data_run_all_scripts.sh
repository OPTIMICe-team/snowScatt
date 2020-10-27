#! /bin/bash

wget https://zenodo.org/record/4118243/files/data_snowScatt.tgz
tar -xzvf data_snowScatt.tgz

mkdir -p data
wget https://github.com/rhoneyager/scatdb/raw/master/share/scatdb.csv
mkdir -p data/tables
mv scatdb.csv data/tables

wget https://zenodo.org/record/1341390/files/tripex_joy_tricr00_l2_any_v00_20151124000000.nc
mkdir -p data/P3
mv tripex_joy_tricr00_l2_any_v00_20151124000000.nc data/P3/

python3 Fig1_ssrga_beta_gamma_asymptote.py
python3 Fig3_plot_agg_shape.py
python3 Fig4_particle_properties.py
python3 Fig5_6_ssrga_evaluation.py   
python3 Fig7_ensemble_subsampling.py  
python3 Fig8_9_P3_application.py