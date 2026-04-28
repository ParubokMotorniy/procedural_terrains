#!/bin/sh

if ! [ -d ../datasets/generated_heightmaps ]; then
    echo "The directory with synthetic heightmaps does not exist! Pull this dataset from the drive mentioned in README first."
    return
fi

echo "Installing pip dependencies!"
pip install -r ./requirements.txt

#trains the rating models based on the precomputed vectors. Omit `--use-saved` and pass paths to directories with boring&interesting `.jpg` heightmaps to recompute the metric vectors anew.
python3 ./model_trainer_and_bulk_metric_computer.py --format jpg --chunk-size 4 --division-depth 4 --mode train  --directory-interesting ./precomputed_metric_vectors/interesting_real_terrains/ --directory-boring ./precomputed_metric_vectors/boring_real_terrains/ --use-saved &> train.log

#tests how well the metrics separate the real-world data based on precomputed metric vectors. Again, if you decide to recompute the metric vectors - pass the paths to new csv-s here.
python3 ./data_separability_evaluator.py --vectors-interesting ./precomputed_metric_vectors/interesting_real_terrains/interesting_metric_vectors_4x4.csv --vectors-boring ./precomputed_metric_vectors/boring_real_terrains/boring_metric_vectors_4x4.csv --filename-tag real_data  &> evaluate_real.log 

#tests how well the metrics separate the real-world data based on precomputed metric vectors. Again, if you decide to recompute the metric vectors - pass the paths to new csv-s here.
python3 ./data_separability_evaluator.py --vectors-interesting ./precomputed_metric_vectors/interesting_llm_terrains/interesting_metric_vectors_4x4.csv --vectors-boring ./precomputed_metric_vectors/boring_llm_terrains/boring_metric_vectors_4x4.csv --filename-tag llm_data  &> evaluate_llm.log 

# Commands below run rating of the synthetic heightmaps by both MLP and LLM. Note that you'll need to provide your own Gemini API key in `llm_baseline_provider.py`
# You can replace `--custom-model-path .` with `--custom-model-path ./models/`.

#Custom model + fine-tuned
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/FFT       --rmd-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/RMD       --sdf-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/SDF       --un-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/UN       --class-mode custom --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of fine-tuned heightmaps" --sub-title "Erosion: None. Model: MLP."  &> bench_custom_no_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/SDF --un-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/UN --class-mode custom --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of fine-tuned heightmaps" --sub-title "Erosion: PE. Model: MLP."  &> bench_custom_pe_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/SDF --un-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/UN --class-mode custom --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of fine-tuned heightmaps" --sub-title "Erosion: CE. Model: MLP."  &> bench_custom_ce_erosion.log &

#LLM + fine-tuned
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/FFT       --rmd-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/RMD       --sdf-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/SDF       --un-dir ../datasets/generated_heightmaps/fine_tuned/no_erosion/UN       --class-mode llm    --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of fine-tuned heightmaps" --sub-title "Erosion: None. Model: LLM." &> bench_llm_no_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/SDF --un-dir ../datasets/generated_heightmaps/fine_tuned/particle_erosion/UN --class-mode llm    --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of fine-tuned heightmaps" --sub-title "Erosion: PE. Model: LLM."   &> bench_llm_pe_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/SDF --un-dir ../datasets/generated_heightmaps/fine_tuned/cellular_erosion/UN --class-mode llm    --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of fine-tuned heightmaps" --sub-title "Erosion: CE. Model: LLM."   &> bench_llm_ce_erosion.log &

# Custom model + randomized
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/randomized/no_erosion/FFT       --rmd-dir ../datasets/generated_heightmaps/randomized/no_erosion/RMD       --sdf-dir ../datasets/generated_heightmaps/randomized/no_erosion/SDF       --un-dir ../datasets/generated_heightmaps/randomized/no_erosion/UN       --class-mode custom --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of randomized heightmaps" --sub-title "Erosion: None. Model: MLP." &> random_bench_custom_no_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/randomized/particle_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/randomized/particle_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/randomized/particle_erosion/SDF --un-dir ../datasets/generated_heightmaps/randomized/particle_erosion/UN --class-mode custom --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of randomized heightmaps" --sub-title "Erosion: PE. Model: MLP."    &> random_bench_custom_pe_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/SDF --un-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/UN --class-mode custom --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of randomized heightmaps" --sub-title "Erosion: CE. Model: MLP."   &> random_bench_custom_ce_erosion.log &

# LLM + randomized
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/randomized/no_erosion/FFT       --rmd-dir ../datasets/generated_heightmaps/randomized/no_erosion/RMD       --sdf-dir ../datasets/generated_heightmaps/randomized/no_erosion/SDF       --un-dir ../datasets/generated_heightmaps/randomized/no_erosion/UN       --class-mode llm    --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of randomized heightmaps" --sub-title "Erosion: None. Model: LLM."  &> random_bench_llm_no_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/randomized/particle_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/randomized/particle_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/randomized/particle_erosion/SDF --un-dir ../datasets/generated_heightmaps/randomized/particle_erosion/UN --class-mode llm    --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of randomized heightmaps" --sub-title "Erosion: PE. Model: LLM."   &> random_bench_llm_pe_erosion.log &
python3 ./synthetic_aesthetics_evaluation.py --fft-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/FFT --rmd-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/RMD --sdf-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/SDF --un-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/UN --class-mode llm    --custom-model-path . --division-depth 4 --chunk-size 4 --plot-title "Aesthetics evaluation of randomized heightmaps" --sub-title "Erosion: CE. Model: LLM."   &> random_bench_llm_ce_erosion.log &

# Commands below build performance plots

python3 ./synthetic_log_speed_evaluation.py --data-dir ../datasets/generated_heightmaps/randomized/no_erosion/UN  --stopwatch-freq 10000000 --plot-title "UN performance evaluation"  &
python3 ./synthetic_log_speed_evaluation.py --data-dir ../datasets/generated_heightmaps/randomized/no_erosion/SDF --stopwatch-freq 10000000 --plot-title "SDF performance evaluation" &
python3 ./synthetic_log_speed_evaluation.py --data-dir ../datasets/generated_heightmaps/randomized/no_erosion/FFT --stopwatch-freq 10000000 --plot-title "FFT performance evaluation" &
python3 ./synthetic_log_speed_evaluation.py --data-dir ../datasets/generated_heightmaps/randomized/no_erosion/RMD --stopwatch-freq 10000000 --plot-title "RMD performance evaluation" &

python3 ./synthetic_log_speed_evaluation.py --data-dir ../datasets/generated_heightmaps/randomized/cellular_erosion/SDF --stopwatch-freq 10000000 --plot-title "CE performance evaluation"  & 
python3 ./synthetic_log_speed_evaluation.py --data-dir ../datasets/generated_heightmaps/randomized/particle_erosion/SDF --stopwatch-freq 10000000 --plot-title "PE performance evaluation" --resolutions "64 128 256 512 1024"  &

python3 ./synthetic_aggregate_speed_evaluation.py --data-dirs "../datasets/generated_heightmaps/randomized/no_erosion/UN ../datasets/generated_heightmaps/randomized/no_erosion/SDF ../datasets/generated_heightmaps/randomized/no_erosion/RMD ../datasets/generated_heightmaps/randomized/no_erosion/FFT" --stopwatch-freq 10000000 --plot-title "Side-by-side performance evaluation" --algos-names "UN SDF RMD FFT" --x-axis-title "Heightmap side size (texels)" --x-axis-ticks "32 64 128 256 512"
