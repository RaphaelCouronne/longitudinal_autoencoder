NUM=1000000
DIM=5
EPOCHS=100
FOLDER="results/FIG3"
DATASET=ADNI_COG
VERSION=cog
W_SPEARMAN=0.1
BATCH=32
LR=0.01
W_ATT=20
KAPPA=0.065
KAPPAMAX=0.08
VER=33
PIMODE=max

for CV_INDEX in 0 1 2 3 4
do
    # LSSL
    python main.py --model_name VaeLSSL --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --w_att $W_ATT --use_GECO --cv_index $CV_INDEX --kappa $KAPPA --verbose $VER

    # Ours
    python main.py --model_name LongVAE --pi_mode $PIMODE --use_softrank --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR  --w_att $W_ATT --use_GECO --cv_index $CV_INDEX --kappa $KAPPA --verbose $VER
done