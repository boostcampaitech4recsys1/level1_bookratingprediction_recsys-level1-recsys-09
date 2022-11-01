models="FM FFM NCF WDN DCN"

for model in ${models}
do
    python main.py \
        --MODEL ${model} \
        --DEVICE cuda \
        --EPOCHS 30
done

deep_models="CNN_FM DeepCoNN"

for d_model in ${deep_models}
do
    python main.py \
        --MODEL ${d_model} \
        --DEVICE cuda \
        --EPOCHS 100
done