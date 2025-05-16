for i in MAVEN; do
    # gán class-num theo dataset
    if [ "$i" = "ACE" ]; then
        m=10
    else
        m=20
    fi

    # lặp qua shot-num = 5 và 10
    for j in 5 10; do
        for k in 42 1 2 3 4; do
            python train.py \
                --data-root ./augmented_data \
                --stream-root ./augmented_data \
                --dataset "$i" \
                --backbone bert-base-uncased \
                --seed $k \
                --lr 2e-5 \
                --decay 1e-4 \
                --no-freeze-bert \
                --shot-num "$j" \
                --batch-size 16 \
                --device cuda:0 \
                --log \
                --log-dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                --log-name "${i}_${j}shot" \
                --dweight_loss \
                --rep-aug mean \
                --distill mul \
                --epochs 30 \
                --class-num "$m" \
                --single-label \
                --cl-aug shuffle \
                --aug-repeat-times 5 \
                --joint-da-loss none \
                --sub-max \
                --cl_temp 0.07 \
                --tlcl \
                --ucl \
                --skip-first-cl ucl+tlcl \
                --sam \
                --sam-type current \
                --rho 0.05
        done
    done
done
