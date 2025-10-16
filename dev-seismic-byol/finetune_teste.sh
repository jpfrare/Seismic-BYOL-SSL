python new_cli_finetune.py \
    --pretrain_data coco \
    --finetune_data seam_ai_N \
    --repetition 10 \
    --layer_lrs '{"stem":0.0, "layer1":1e-6, "layer2":1e-5, "layer3":1e-4, "layer4":1e-3}' \
    --cap 1.0