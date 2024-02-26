epoch=2
model_name='bert-base-cased'
model_type='bert'
for lr in 0.00005
    do
    for train_id in  0 1 2 3 4
        do
        for train_mode in 'null' 'OM' 'OC'
            do
            for batch_size in   32
            do
            
                dataset_name="conll2003"
                echo "dataset:$dataset_name"
                echo "train_id:$train_id"
                echo "train_mode":$train_mode
                echo "learning_rate:$lr"
                echo "epoch:$epoch"
                python -u trainer.py --gpu_id 3 \
                --train_id $train_id \
                --data_dir $dataset_name'/data/' \
                --pad_to_max_length  \
                --model_name_or_path $model_name \
                --model_type $model_type \
                --tokenizer_name $model_name \
                --num_train_epochs $epoch \
                --per_device_train_batch_size $batch_size \
                --per_device_eval_batch_size  32 \
                --max_train_steps -1 \
                --output_dir $dataset_name'/model/' \
                --seed 1 \
                --model_mode $train_mode \
                --learning_rate $lr
               
            done
            done
        done
    done