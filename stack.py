import sys
import numpy as np
from diversity_utils import *
from distance import *

from numpy import dot
from numpy.linalg import norm
import pandas as pd



from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import StackingClassifier


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


import argparse

parser = argparse.ArgumentParser(description='Train model with fault params')
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'gtsrb', 'pneumonia'], default='mnist')
parser.add_argument('--final_fault', type=str, default="golden")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--natural', action='store_true')
parser.add_argument('--ens_size', type=int, default=3)

parser.add_argument('--modelA', type=str, default="ConvNet")
parser.add_argument('--modelB', type=str, default="DeconvNet")
parser.add_argument('--modelC', type=str, default="VGG11")

args = parser.parse_args()

dataset = args.dataset
final_fault = args.final_fault
num_epochs = args.epochs
batch_size = args.batch_size

modelA_name = args.modelA
modelB_name = args.modelB
modelC_name = args.modelC

ens_size = args.ens_size


def generate_csv(y_test, predictions_A):
    list_name = ['ground_truth']
    df = pd.DataFrame(columns=list_name)

    df["ground_truth"] = y_test
    df["predicted_A"] = predictions_A

    file_name = "./remix_results/stack-" + str(ens_size)  + "_" + modelA_name + "_" + dataset + "_" +  final_fault + ".csv"
    df.to_csv(file_name)


def main(argv):
    symmetric = not args.natural

    (x_train, y_train), (x_test, y_test) = load_training_data(dataset, final_fault, symmetric)


    input_shape = x_train.shape[1:]
    modelA = get_model_by_name(modelA_name, input_shape)
    modelB = get_model_by_name(modelB_name, input_shape)
    modelC = get_model_by_name(modelC_name, input_shape)

    modelA_estimator = KerasClassifier(build_fn= modelA, optimizer=tf.keras.optimizers.Adam(), epochs=num_epochs, batch_size=batch_size, verbose=1)
    modelB_estimator = KerasClassifier(build_fn= modelB, optimizer=tf.keras.optimizers.Adam(), epochs=num_epochs, batch_size=batch_size, verbose=1)
    modelC_estimator = KerasClassifier(build_fn= modelC, optimizer=tf.keras.optimizers.Adam(), epochs=num_epochs, batch_size=batch_size, verbose=1)

    est_list = [("modelA", modelA_estimator), ("modelB", modelB_estimator), ("modelC", modelC_estimator)]

    boosted_ann = StackingClassifier(estimators=est_list, cv=2)
    boosted_ann.fit(x_train, y_train) 

    predictions_A = boosted_ann.predict(x_test)
    y_test = y_test.flatten()

    generate_csv(y_test, predictions_A)


if __name__ == "__main__":
    main(sys.argv[1:])

