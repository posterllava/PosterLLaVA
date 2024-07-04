ckpt_name=posterllava_v0

python llava/serve/cli_multi.py \
--model-path pretrained_model/$ckpt_name \
--json-file data/qbposter/qbposter_val_instruct.json \
--output-file output/${ckpt_name}_output_val.json \
--num-gpus 1 --data-path ./data/