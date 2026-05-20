# Craft 10 1-to-1 BadNet models both original and Mixed-label versions with Dirty Label Poisoning Rate = 1% (of the source class data).

# Please make sure that the attack is successful before deploying defense.

echo ----------------------------start training ------------------------------
for (( integer = 0; integer <= 9; integer++ ))
do
    python attacks_crafting.py     --out_dir   attack/attack$integer --DPR=0.01 --ratio=2
    python train_models_contam_ml.py --attack_dir attack/attack$integer --model_dir attack/model$integer --DPR=0.01 --ratio=2
    python train_models_contam.py    --attack_dir attack/attack$integer --model_dir attack/model$integer
done
