# NER Data Bias
## build data

```shell
    dataset_name = "conll2003"  
    python  data_process/dataset_split.py --dataset_name $dataset_name  \
              --data_path $dataset_name'/data/train.json' \
              --output_path  $dataset_name'/data/' \
              --split_number 5 \
              --tokenizer_path "bert-base-cased"
              --tokenizer_name "bert"
```
after this step, you will see 5(split_number) train/test file in folder: ${dataset}/data 

## train model

```shell
    sh script/run_trainer.sh
```
after training, you will see context/entity/null best-dev-checkpoint in folder: ${dataset}/model

## Calculate data PVI

```shell
    sh script/run_V_info.sh
```
after training, you will see the pvi value for context/entity in folder: ${dataset}/PVI

## split dataset by context/entity PVI (High CEIM/Low CEIM/near-zero CEIM)
```shell
    dataset_name = "conll2003"  
    python  data_process/select_data_by_pvi.py --dataset_name $dataset_name  \
              --split_number 5 \
              --tokenizer_path "bert-base-cased"
              --tmodel_type "bert"
```

## Reducing the V-information of Entity
-  Random2low
```shell
    model_type='bert'
    model_name='bert-base-cased'
    gpu_id=0
    dataset='conll2003'
    change_type="aug"
    aug_om_small_mode="None"
    aug_mode="om_small"
    seed=1

    for aug_rate in   0.2 0.4
        do       
            data_name='train'    
            om_small_file_name='OM_smaller_total_entity.json'
            out_dir=$dataset"/model/"

            python3 -u bert_tagger_aug.py --data_dir "./"$dataset"/data/" \
            --om_small_file_name $om_small_file_name \
            --change_type $change_type \
            --aug_mode $aug_mode \
            --aug_rate $aug_rate \
            --aug_om_small_mode $aug_om_small_mode \
            --model_type $model_type \
            --dataset_name $dataset \
            --model_name_or_path $model_name \
            --output_dir $out_dir \
            --overwrite_output_dir \
            --max_seq_length  128 \
            --logging_steps -1 \
            --evaluate_during_training \
            --local_rank  0 \
            --train_mode $data_name \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 16 \
            --gpu_id $gpu_id \
            --learning_rate 0.00005 \
            --save_steps -1 \
            --seed $seed \
            --num_train_epochs 7 \
            --do_predict \
            --do_train \
            --do_eval \
            
            done
```
- HighC2low

```shell
    model_type='bert'
    model_name='bert-base-cased'
    gpu_id=0
    dataset='conll2003'
    change_type="aug"
    aug_om_small_mode="select_oc_bigger"
    aug_mode="om_small"
    seed=1

    for aug_rate in   0.2 0.4
        do       
            data_name='train'    
            om_small_file_name='OM_smaller_total_entity.json'
            out_dir=$dataset"/model/"

            python3 -u bert_tagger_aug.py --data_dir "./"$dataset"/data/" \
            --om_small_file_name $om_small_file_name \
            --change_type $change_type \
            --aug_mode $aug_mode \
            --aug_rate $aug_rate \
            --aug_om_small_mode $aug_om_small_mode \
            --model_type $model_type \
            --dataset_name $dataset \
            --model_name_or_path $model_name \
            --output_dir $out_dir \
            --overwrite_output_dir \
            --max_seq_length  128 \
            --logging_steps -1 \
            --evaluate_during_training \
            --local_rank  0 \
            --train_mode $data_name \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 16 \
            --gpu_id $gpu_id \
            --learning_rate 0.00005 \
            --save_steps -1 \
            --seed $seed \
            --num_train_epochs 7 \
            --do_predict \
            --do_train \
            --do_eval \
            
            done
```

- Redundant2low

```shell

    model_type='bert'
    model_name='bert-base-cased'
    gpu_id=0
    dataset='conll2003'
    change_type="replace"
    aug_om_small_mode="None"
    aug_mode="om_small"
    seed=1

    python robust_data_entity.py  --dataset_name $dataset \
          --tokenizer_path "bert-base-cased" \
          --model_type $model_type 
    
    for aug_rate in   0.2 0.4
        do       
            data_name='train'    
            om_small_file_name='OM_smaller_total_entity.json'
            out_dir=$dataset"/model/"

            python3 -u bert_tagger_aug.py --data_dir "./"$dataset"/data/" \
            --om_small_file_name $om_small_file_name \
            --change_type $change_type \
            --aug_mode $aug_mode \
            --aug_rate $aug_rate \
            --aug_om_small_mode $aug_om_small_mode \
            --model_type $model_type \
            --dataset_name $dataset \
            --model_name_or_path $model_name \
            --output_dir $out_dir \
            --overwrite_output_dir \
            --max_seq_length  128 \
            --logging_steps -1 \
            --evaluate_during_training \
            --local_rank  0 \
            --train_mode $data_name \
            --per_gpu_train_batch_size 16 \
            --per_gpu_eval_batch_size 16 \
            --gpu_id $gpu_id \
            --learning_rate 0.00005 \
            --save_steps -1 \
            --seed $seed \
            --num_train_epochs 7 \
            --do_predict \
            --do_train \
            --do_eval \
            
            done
```

## Enhancing the V-information of Context

- Random2High
```shell
    dataset_name = "conll2003"  
    python  robust_data_context.py  --dataset_name $dataset_name  \
              --reduction_rate 0.2 \
              --oc_select_mode "random"
```
- Low2High
```shell
    dataset_name = "conll2003"  
    python  robust_data_context.py  --dataset_name $dataset_name  \
              --reduction_rate 0.2 \
              --oc_select_mode "small"
```
