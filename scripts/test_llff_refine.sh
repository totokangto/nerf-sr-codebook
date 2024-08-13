dataset="fern"
W=504
H=378
# batch_size=32
downscale=2
batch_size=1
option="patchsamearea8"

data_root='/data/csj000714/data'

python test_refine.py --name llff-refine-${dataset}-${H}x${W}-ni-dp-ds${downscale}-${option} \
    --dataset_mode llff_refine --dataset_root ${data_root}/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/nerf-sr-refine/ --summary_dir ./logs/nerf-sr-refine --results_dir ./results/nerf-sr-refine \
    --img_wh $W $H --batch_size $batch_size \
    --model refine --test_split test_train \
    --syn_dataroot ./checkpoints/nerf-sr/llff-${dataset}-${H}x${W}-ni64-dp-ds${downscale}/30_val_vis 

    # --test_split test --refine_network maxpoolingmodel --load_epoch 3 \