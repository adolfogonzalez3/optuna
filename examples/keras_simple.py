"""
Optuna example that optimizes multi-layer perceptrons using Keras.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Keras and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python keras_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize keras_simple.py objective --n-trials=100 \
      --study $STUDY_NAME --storage sqlite:///example.db

"""

from __future__ import division
from __future__ import print_function


import os
import numpy as np

# Caveat: The theano backend must be greater than v1.0.4 due to changes in
# NumPy.
# Setting BACKEND in script
os.environ['KERAS_BACKEND'] = 'tensorflow'

import optuna
import keras

try:
    import tensorflow
    TENSORFLOW_BACKEND = True
except ModuleNotFoundError:
    TENSORFLOW_BACKEND = False

N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_network(trial, features):
    '''Create keras model.'''
    # We optimize the numbers of layers and their units.
    input_layer = keras.Input(shape=features)
    prev_layer = keras.layers.Flatten()(input_layer)

    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        n_units = trial.suggest_int('n_units_l{}'.format(i), 1, 128)
        prev_layer = keras.layers.Dense(
            units=n_units, activation='relu'
        )(prev_layer)

    logits = keras.layers.Dense(units=10, activation='softmax')(prev_layer)
    return keras.Model(inputs=input_layer, outputs=logits)


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = keras.optimizers.Adam(lr=adam_lr, decay=weight_decay)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        momentum = trial.suggest_loguniform('momentum', 1e-5, 1e-1)
        optimizer = keras.optimizers.SGD(
            lr=momentum_sgd_lr, momentum=momentum, decay=weight_decay
        )

    return optimizer


def sample(data, labels, size):
    '''Randomly sample from data set.'''
    indices = np.random.choice(np.arange(len(data)), size=size, replace=False)
    return data[indices], labels[indices]


def objective(trial):
    # Reset session if backend is tensorflow
    if TENSORFLOW_BACKEND:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
        keras.backend.clear_session()
    train_dataset, eval_dataset = keras.datasets.mnist.load_data()
    train_data, train_labels = sample(*train_dataset, size=N_TRAIN_EXAMPLES)
    eval_data, eval_labels = sample(*eval_dataset, size=N_TEST_EXAMPLES)

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)

    mnist_classifier = create_network(trial, train_data.shape[1:])
    mnist_classifier.compile(
        optimizer=create_optimizer(trial),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    mnist_classifier.fit(
        x=train_data, y=train_labels, batch_size=BATCHSIZE, shuffle=True,
        epochs=EPOCH, verbose=0
    )

    _, accuracy = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, batch_size=BATCHSIZE, verbose=0
    )
    return float(accuracy)


def main():
    # Reduce messages from tensorflow if backend is tensorflow
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == "__main__":
    main()
