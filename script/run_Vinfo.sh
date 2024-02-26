for dataset_name in 'conll2003'
    do
    for train_id in 0 1 2 3 4 
        do
        temp_dataset_dir=$dataset_name'/'$mode
        python v_info.py --data_dir $temp_dataset_dir'/data/' \
                        --model_dir $temp_dataset_dir'/model/' \
                        --model_type 'bert' \
                        --dataset_name $temp_dataset_dir \
                        --tokenizer_name 'bert-base-cased' \
                        --gpu_id  3 \
                        --train_id $train_id
        done
      
    done