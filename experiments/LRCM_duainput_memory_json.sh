#!/bin/bash

checkpoint=ckpt/nar_stage2.ckpt

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=data/Multimodel_Text_dataset_updating/testset
wav_dir=data/Multimodel_Text_dataset_updating/testset
basenames=$(cat "${data_dir}/gen_files.txt")

json_file="${data_dir}/gen_texts.json"

start=0
seed=43
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

    audio_key="${wavfile}"

    if command -v jq &> /dev/null; then
        global_text=$(jq -r ".\"${audio_key}\".global // empty" "${json_file}")
        local_text=$(jq -r ".\"${audio_key}\".local // empty" "${json_file}")

        if [[ -n "$global_text" && -n "$local_text" ]]; then
            text="${global_text}, ${local_text}"
        elif [[ -n "$global_text" ]]; then
            text="$global_text"
        elif [[ -n "$local_text" ]]; then
            text="$local_text"
        else
            text=""
        fi
    else
        echo "jq not found, using default text"
        text=""
    fi

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


{
echo "# Generation setting saving - $(date)"
echo "checkpoint=${checkpoint}"
echo "dest_dir=${dest_dir}"
echo "data_dir=${data_dir}"
echo "wav_dir=${wav_dir}"
echo "json_file=${json_file}"
echo "start=${start}"
echo "seed=${seed}"
echo "fps=${fps}"
echo "trim_s=${trim_s}"
echo "length_s=${length_s}"
echo "trim=${trim}"
echo "length=${length}"
echo "fixed_seed=${fixed_seed}"
echo "gpu=${gpu}"
echo "render_video=${render_video}"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "git_branch=$(git branch --show-current 2>/dev/null || echo 'unknown')"
} > "${dest_dir}/experiment_config.log"

cp "${json_file}" "${dest_dir}/" 2>/dev/null
cp "$0" "${dest_dir}/script.sh"

echo "The experiment configuration has been saved to ${dest_dir}/experiment_config.log"
