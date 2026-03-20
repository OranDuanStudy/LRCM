#!/bin/bash

checkpoint=ckpt/ar_stage3.ckpt
dest_dir=result/generated/dual_clip_ddpm_1114stage3_epoch3_fullprompts

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=data/Multimodel_Text_dataset_updating/testset
wav_dir=data/Multimodel_Text_dataset_updating/testset
basenames=$(cat "${data_dir}/gen_files.txt")

readarray -t prompts < "${data_dir}/gen_texts.txt"

start=0
seed=150
fps=30
trim_s=0
length_s=400
trim=$((trim_s*fps))
length=$((length_s*fps))
fixed_seed=false
gpu="cuda:0"
render_video=true
segment_frames=300

file_index=0

for wavfile in $basenames; 
do
	start=0
	
    text="${prompts[$file_index]}"

	echo "Processing file: ${wavfile}, with prompt: ${text}"
	for postfix in 0
	do
		input_file=${wavfile}.audio29_${fps}fps.pkl
		
		output_file=${wavfile::-3}
		
		echo "start=${start}, len=${length}, postfix=${postfix}, seed=${seed}"
		python synthesize.py --checkpoints="${checkpoint}" --data_dirs="${data_dir}" --input_files="${input_file}" --input_text="${text}" --start=${start} --end=${length} --seed=${seed} --postfix=${postfix} --trim=${trim} --dest_dir=${dest_dir} --gpu=${gpu} --video=${render_video} --outfile=${output_file} --segment-frames=${segment_frames}
		if [ "$fixed_seed" != "true" ]; then
			seed=$((seed+1))
		fi 
		echo seed=$seed
		python utils/cut_wav.py ${wav_dir}/${wavfile::-3}.wav $(((start+trim)/fps)) $(((start+length-trim)/fps)) ${postfix} ${dest_dir}
		if [ "$render_video" == "true" ]; then
			ffmpeg -y -i ${dest_dir}/${output_file}.mp4 -i ${dest_dir}/${wavfile::-3}_${postfix}.wav ${dest_dir}/${output_file}_audio.mp4
			rm ${dest_dir}/${output_file}.mp4
		fi
		
		start=$((start+length))
	done
	
	file_index=$((file_index + 1))
done
