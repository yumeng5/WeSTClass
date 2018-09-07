# WeSTClass

The source code used for Weakly-Supervised Neural Text Classification, published in CIKM 2018.

## Requirments

Before running, you need to first install the required packages by typing following commands:

```
$ pip3 install -r requirements.txt
```

## Running

```
python main.py --dataset ${dataset} --sup_source ${sup_source} --model ${model}
```
where you need to specify the dataset in ```${dataset}```, the weak supervision type in ```${sup_source}``` (could be one of ```['labels', 'keywords', 'docs']```), and the type of neural model to use in ```${model}``` (could be one of ```['cnn', 'rnn']```).

An example run is provided in ```test.sh```, which can be executed by 
```
./test.sh
```

More advanced settings on training and hyperparameters are commented in ```main.py```.

## Inputs
The weak supervision sources ```${sup_source}``` can come from any of the following:
* Label surface names (```labels```); you need to provide class names for each class in ```/${dataset}/classes.txt```, where 
* Class-related keywords (```keywords```);
* Labeled documents (```docs```);



## Citations
