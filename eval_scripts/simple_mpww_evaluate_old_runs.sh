export TOKENIZERS_PARALLELISM=false


./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/large_bs_runs/all_datasets_bs2560_hard_neg-2022-01-13_04-09-51/checkpoints all_bs2560_hardneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/large_bs_runs/kelm_bs2560_hard_neg-2022-01-13_04-22-36/checkpoints kelm_bs2560_hardneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/large_bs_runs/tekgen_bs2560_hard_neg-2022-01-13_05-28-18/checkpoints tekgen_bs2560_hardneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/large_bs_runs/trex_bs2560_hard_neg-2022-01-13_05-44-27/checkpoints trex_bs2560_hardneg

./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs/all_datasets_bs320-2022-01-11_02-41-31/checkpoints all_bs320_batchneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs/kelm_bs320-2022-01-11_06-51-27/checkpoints kelm_bs320_batchneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs/tekgen_bs320-2022-01-11_13-19-54/checkpoints tekgen_bs320_batchneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs/trex_bs320-2022-01-11_10-21-36/checkpoints trex_bs320_batchneg

./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs_hard_neg/all_datasets_bs192_hard_neg-2022-01-12_20-02-34/checkpoints all_bs192_hardneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs_hard_neg/kelm_bs192_hard_neg-2022-01-12_03-25-06/checkpoints kelm_bs192_hardneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs_hard_neg//tekgen_bs192_hard_neg-2022-04-07_22-26-40/checkpoints tekgen_bs192_hardneg
./eval_scripts/eval_simple_mpww.sh /home/teven_huggingface_co/tetraencoder/outputs/small_bs_runs_hard_neg/trex_bs192_hard_neg-2022-01-13_04-19-46/checkpoints trex_bs192_hardneg

./eval_scripts/eval_simple_mpww.sh outputs/baseline/ baseline
