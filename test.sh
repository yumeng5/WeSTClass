
dataset=agnews
sup_source=labels
model=cnn

export CUDA_VISIBLE_DEVICES=0

python main.py --dataset ${dataset} --sup_source ${sup_source} --model ${model} --with_evaluation True
