#! /bin/bash

# ./predict.sh ara_seq eff_seq ./tmp/test_jobid/
python ./RF_esm_net/predict_step1_map_input_to_tair.py --ara_seq $1 --eff_seq  $2 --outdir $3

    
conda_init(){
	# >>> conda initialize >>>
	# !! Contents within this block are managed by 'conda init' !!
	__conda_setup="$('/home/v/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
	if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
	else
	    if [ -f "/home/v/miniconda3/etc/profile.d/conda.sh" ]; then
		. "/home/v/miniconda3/etc/profile.d/conda.sh"
	    else
		export PATH="/home/v/miniconda3/bin:$PATH"
	    fi
	fi
	unset __conda_setup
	# <<< conda initialize <<<
}

conda_init
conda activate esm
python RF_esm_net/predict_step2_generate_esm_feature.py --input $3/cache/excluded_ara.fasta --outdir $3

python RF_esm_net/predict_step3.py --ppis $3/ppi_list.txt --add_esm_feature $3/exclude_esm_mean.pkl --outdir $3
