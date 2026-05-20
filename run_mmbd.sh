# Run MMBD-CSO on the attacked models
echo ---------------------------- start mmbd_cso detection ------------------------------
for (( integer = 0; integer <= 9; integer++ ))
do
    echo ---------------------------- BadNet_ML index $integer ------------------------------
    python ../mmbd_cso.py \
        --ckpt_path attack/model$integer/model_ml_0.01_2.pth

    echo ---------------------------- BadNet index $integer ------------------------------
    python ../mmbd_cso.py \
        --ckpt_path attack/model$integer/model.pth
done
