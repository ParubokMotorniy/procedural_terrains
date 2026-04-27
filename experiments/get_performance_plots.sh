#!/bin/sh

python3 ../synthetic_log_speed_evaluation.py --data-dir ../generated_heightmaps/randomized/no_erosion/UN  --stopwatch-freq 10000000 --plot-title "UN performance evaluation"  &
python3 ../synthetic_log_speed_evaluation.py --data-dir ../generated_heightmaps/randomized/no_erosion/SDF --stopwatch-freq 10000000 --plot-title "SDF performance evaluation" &
python3 ../synthetic_log_speed_evaluation.py --data-dir ../generated_heightmaps/randomized/no_erosion/FFT --stopwatch-freq 10000000 --plot-title "FFT performance evaluation" &
python3 ../synthetic_log_speed_evaluation.py --data-dir ../generated_heightmaps/randomized/no_erosion/RMD --stopwatch-freq 10000000 --plot-title "RMD performance evaluation" &

python3 ../synthetic_log_speed_evaluation.py --data-dir ../generated_heightmaps/randomized/cellular_erosion/SDF --stopwatch-freq 10000000 --plot-title "CE performance evaluation"  & 
python3 ../synthetic_log_speed_evaluation.py --data-dir ../generated_heightmaps/randomized/particle_erosion/SDF --stopwatch-freq 10000000 --plot-title "PE performance evaluation"  &
