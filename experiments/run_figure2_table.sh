NUM=5000
DIM=5
EPOCHS=101
FOLDER="results/FIG2"
DATASET=STARMEN
VERSION=64
W_SPEARMAN=0.1
BATCH=32
LR=0.01
RANDOMSELECT=3
LOGVARMIN=-5
NW=3
PIMODE=max
VER=33
INPUT_FOLER="/network/lustre/dtlake01/aramis/projects/collaborations/UnsupervisedLongitudinal_IPMI21/synthetic/"

for CV_INDEX in 0 1 2 3 4
do
    # long VAE Tests
    python main.py --dataset_input_path $INPUT_FOLER --model_name LongVAE --pi_mode $PIMODE --random_select $RANDOMSELECT --use_softrank --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX  --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER
    # No Soft Ranking
    python main.py --dataset_input_path $INPUT_FOLER --model_name LongVAE --pi_mode $PIMODE --random_select $RANDOMSELECT --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX  --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER
    # No DeepSet
    python main.py --dataset_input_path $INPUT_FOLER --model_name LongVAE --pi_mode identity --random_select $RANDOMSELECT --use_softrank --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX  --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER
    # No DeepSet 1 Encoder
    python main.py --dataset_input_path $INPUT_FOLER --model_name LongVAE --pi_mode identity --random_select $RANDOMSELECT --use_softrank --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX  --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER --one_encoder
done

for CV_INDEX in 0 1 2 3 4
do
    # BVAE
    python main.py --dataset_input_path $INPUT_FOLER --model_name BVAE --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER

    # BVAE 10
    python main.py --dataset_input_path $INPUT_FOLER --model_name BVAE --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER --w_kl 10

    # BVAE_Regression
    python main.py --dataset_input_path $INPUT_FOLER --model_name BVAE_Regression --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR  --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER

    # LSSL
    python main.py --dataset_input_path $INPUT_FOLER --model_name VaeLSSL --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER

    # MLVAE
    python main.py --dataset_input_path $INPUT_FOLER --model_name MLVAE --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER

    # DVAE
    python main.py --dataset_input_path $INPUT_FOLER --model_name DVAE --pi_mode $PIMODE --random_select $RANDOMSELECT --use_softrank --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX  --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER

    # DRVAE
    #python main.py --dataset_input_path $INPUT_FOLER --model_name DRVAE --pi_mode $PIMODE --random_select $RANDOMSELECT --use_softrank --w_spearman $W_SPEARMAN --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER
done

for CV_INDEX in 0 1 2 3 4
do
    # Riemanian VAE
    python main.py --dataset_input_path $INPUT_FOLER --model_name MaxVAE --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER

    # Riemanian AE
    python main.py --dataset_input_path $INPUT_FOLER --model_name MaxAE --dataset $DATASET --dataset_version $VERSION --num_visits $NUM --latent_dimension $DIM --max_epochs $EPOCHS --cuda --folder $FOLDER --batch_size $BATCH --lr $LR --cv_index $CV_INDEX --cliplogvar_min $LOGVARMIN --num_workers $NW  --verbose $VER
done




