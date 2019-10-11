#!/bin/bash 
wget https://www.dropbox.com/s/y8fg5ltc2m3l736/boostingnmodel.pkl?dl=1 -O 'test.pkl'
python3 hw1_best.py $1 $2 test.pkl
