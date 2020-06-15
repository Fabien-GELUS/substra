
## Prerequisites

In order to run this example, you'll need to:

* use Python 3
* have [Docker](https://www.docker.com/) installed
* [install the `substra` cli](../../README.md#install)
* [install the `substratools` library](https://github.com/substrafoundation/substra-tools)
* [pull the `substra-tools` docker images](https://github.com/substrafoundation/substra-tools#pull-from-private-docker-registry)
* create a substra profile to define the substra network to target, for instance:
    ```sh
    substra config --profile node-1 --username node-1 --password 'p@$swr0d44' http://substra-backend.node-1.com
    substra login --profile node-1
    ```

## Data preparation

The first step will be to generate train and test data samples from keras.datasets.mnist


To generate the data samples, run:
```sh
pip install -r scripts/requirements.txt
python scripts/generate_data_samples.py
```

This will create two sub-folders in the `assets` folder:
* `train_data` contains train data features and labels as numpy array files
* `test_data` contains test data features and labels as numpy array files

## Writing the objective and data manager

Both objective and data manager will need a proper markdown description, you can check them out in their respective
folders. Notice that the data manager's description includes a formal description of the data structure.

Notice also that the `metrics.py` and `opener.py` module both rely on classes imported from the `substratools` module.
These classes provide a simple yet rigid structure that will make algorithms pretty easy to write.

## Writing a simple algorithm

You'll find under `assets/algo_cnn` an implementation of the cnn model in the [Keras example](https://keras.io/examples/mnist_cnn/). Like the metrics and opener scripts, it relies on a
class imported from `substratools` that greatly simplifies the writing process. You'll notice that it handles not only
the train and predict tasks but also a lot of data preprocessing.

## Testing our assets

### Using asset command line interfaces

#### Training task

```sh

python3 assets/algo_cnn/algo.py train \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data \
  --output-model-path assets/model/model \
  --log-path assets/logs/train.log


python3 assets/algo_cnn/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data \
  --output-predictions-path assets/pred-train.npy \
  --models-path assets/model/ \
  --log-path assets/logs/train_predict.log \
  model

python3 assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/train_data \
  --input-predictions-path assets/pred-train.npy \
  --output-perf-path assets/perf-train.json \
  --log-path assets/logs/train_metrics.log
  
 ```

#### Testing task

```sh

python3 assets/algo_cnn/algo.py predict \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data \
  --output-predictions-path assets/pred-test.npy \
  --models-path assets/model/ \
  --log-path assets/logs/test_predict.log \
  model
  
python3 assets/objective/metrics.py \
  --debug \
  --opener-path assets/dataset/opener.py \
  --data-samples-path assets/test_data \
  --input-predictions-path assets/pred-test.npy \
  --output-perf-path assets/perf-test.json \
  --log-path assets/logs/test_metrics.log
```

### Using substra cli (wip)

