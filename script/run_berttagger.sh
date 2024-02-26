model_name="bert-base-cased"
model_type="bert"
dataset="conll2003"
bsz=16
lr=0.00005
epoch=7
seed=1

    
    name='replace0.2_train'
    echo "dataset:"$dataset
    echo "lr:"$lr
    echo "bsz:"$bsz

   
    out_dir=$dataset"/model/"
    echo "out_dir:"$out_dir
    python3 -u bert_tagger.py --data_dir ./$dataset/data/replace_context_smaller2bigger/ \
    --model_type $model_type \
    --dataset_name $dataset \
    --model_name_or_path $model_name \
    --output_dir $out_dir \
    --overwrite_output_dir \
    --max_seq_length  128 \
    --logging_steps -1 \
    --evaluate_during_training \
    --local_rank  0 \
    --gpu_id 0 \
    --train_mode $name \
    --per_gpu_train_batch_size $bsz \
    --per_gpu_eval_batch_size 16 \
    --learning_rate $lr \
    --save_steps -1 \
    --num_train_epochs $epoch \
    --seed $seed \
    --do_eval \
    --do_train \
    --do_predict \
   
