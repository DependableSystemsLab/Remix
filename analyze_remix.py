#! /usr/bin/python3

# This script calculates balanced accuracy and f1-score of Remix vs baselines on batch results

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
import pandas as pd
import numpy as np
import math
import statistics



from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score




pd.options.mode.chained_assignment = None



def prep_voting_arr(np_pred, logit_shape):
    voting_arr = np.zeros(logit_shape)
    max_arr = np_pred.reshape(-1,1)
    np.put_along_axis(voting_arr, max_arr, 1, axis=1)
    return voting_arr


def tanh_activation(x):
    n = 20
    return np.tanh(n * x)


def replace_one_with_zero(arr):
    arr[arr==1.] = 0


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)    
    for i in arr:
        temp = (((i - np.min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.asarray(norm_arr)


def individual_model(ens_df, metric="ba"):
    if metric == "f1":
        perfA = f1_score(ens_df["ground_truth"], ens_df["predicted_A"], average="binary")
        perfB = f1_score(ens_df["ground_truth"], ens_df["predicted_B"], average="binary")
        perfC = f1_score(ens_df["ground_truth"], ens_df["predicted_C"], average="binary")
    else:
        perfA = balanced_accuracy_score(ens_df["ground_truth"], ens_df["predicted_A"])
        perfB = balanced_accuracy_score(ens_df["ground_truth"], ens_df["predicted_B"])
        perfC = balanced_accuracy_score(ens_df["ground_truth"], ens_df["predicted_C"])

    perf_vals = [perfA, perfB, perfC]
    return np.average(perf_vals, weights=[perfA/sum(perf_vals), perfB/sum(perf_vals), perfC/sum(perf_vals)])


def static_weighted_voting(ens_df, dataset, total_samples, num_classes):
    logit_shape = (total_samples, num_classes) 
    voting_arr = np.zeros(logit_shape)

    weighted_voting_A = prep_voting_arr(ens_df["predicted_A"].to_numpy(), logit_shape)
    weighted_voting_B = prep_voting_arr(ens_df["predicted_B"].to_numpy(), logit_shape)
    weighted_voting_C = prep_voting_arr(ens_df["predicted_C"].to_numpy(), logit_shape)

    if dataset == "pneumonia":
        modelA_acc = f1_score(ens_df["ground_truth"], ens_df["predicted_A"], average="binary")
        modelB_acc = f1_score(ens_df["ground_truth"], ens_df["predicted_B"], average="binary")
        modelC_acc = f1_score(ens_df["ground_truth"], ens_df["predicted_C"], average="binary")
    else:
        modelA_acc = balanced_accuracy_score(ens_df["ground_truth"], ens_df["predicted_A"])
        modelB_acc = balanced_accuracy_score(ens_df["ground_truth"], ens_df["predicted_B"])
        modelC_acc = balanced_accuracy_score(ens_df["ground_truth"], ens_df["predicted_C"])

    min_acc = min(modelA_acc, modelB_acc, modelC_acc)

    weight_A = modelA_acc - min_acc + 1
    weight_B = modelB_acc - min_acc + 1
    weight_C = modelC_acc - min_acc + 1

    weighted_maj_preds = weight_A * weighted_voting_A + weight_B * weighted_voting_B + weight_C * weighted_voting_C
    ens_df["weighted_maj"] = np.argmax(weighted_maj_preds, axis = 1).flatten()[:total_samples] 


def remix_voting(ens_df, dataset, total_samples, num_classes, logitA, logitB, logitC, distance_metric="corr"):
    logit_shape = (total_samples, num_classes) 
    voting_arr = np.zeros(logit_shape)

    weighted_voting_A = prep_voting_arr(ens_df["predicted_A"].to_numpy(), logit_shape)
    weighted_voting_B = prep_voting_arr(ens_df["predicted_B"].to_numpy(), logit_shape)
    weighted_voting_C = prep_voting_arr(ens_df["predicted_C"].to_numpy(), logit_shape)

    conf_A = ens_df["highest_softmax_A"].to_numpy()
    conf_B = ens_df["highest_softmax_B"].to_numpy()
    conf_C = ens_df["highest_softmax_C"].to_numpy()

    weighted_voting_A = logitA
    weighted_voting_B = logitB
    weighted_voting_C = logitC


    dcorr_AB = distance_metric + "_AB"
    dcorr_BC = distance_metric + "_BC"
    dcorr_AC = distance_metric + "_AC"

    if distance_metric == "cos":
        corr_AB = 1 - ens_df[dcorr_AB].to_numpy()
        corr_BC = 1 - ens_df[dcorr_BC].to_numpy()
        corr_AC = 1 - ens_df[dcorr_AC].to_numpy()
    else:
        corr_AB = ens_df[dcorr_AB].to_numpy()
        corr_BC = ens_df[dcorr_BC].to_numpy()
        corr_AC = ens_df[dcorr_AC].to_numpy()

    ens_size = 3

    weight_A = ((corr_AB + corr_AC) / 2)[:, np.newaxis]
    weight_B = ((corr_AB + corr_BC) / 2)[:, np.newaxis]
    weight_C = ((corr_BC + corr_AC) / 2)[:, np.newaxis]

    sparsity_A = ens_df["sparsity_A"].to_numpy()[:, np.newaxis]
    sparsity_B = ens_df["sparsity_B"].to_numpy()[:, np.newaxis]
    sparsity_C = ens_df["sparsity_C"].to_numpy()[:, np.newaxis]
    
    replace_one_with_zero(sparsity_A)
    replace_one_with_zero(sparsity_B)
    replace_one_with_zero(sparsity_C)

    sparsity_A = tanh_activation(sparsity_A)
    sparsity_B = tanh_activation(sparsity_B)
    sparsity_C = tanh_activation(sparsity_C)


    if distance_metric == "corr" or distance_metric == "cos":
        remix_preds = weighted_voting_A * sparsity_A / weight_A + weighted_voting_B * sparsity_B / weight_B + weighted_voting_C * sparsity_C / weight_C
    elif distance_metric == "w" or distance_metric == "frob":
        remix_preds = weighted_voting_A * sparsity_A * weight_A + weighted_voting_B * sparsity_B * weight_B + weighted_voting_C * sparsity_C * weight_C


    ens_df["remix"] = np.argmax(remix_preds, axis = 1).flatten()[:total_samples]

    ens_df["remix_max"] = np.max(remix_preds, axis = 1).flatten()[:total_samples]
    ens_df["remix_sum"] = np.sum(remix_preds, axis = 1).flatten()[:total_samples]

    ens_df["remix_avg"] = ens_df["remix_max"] / ens_df["remix_sum"]


def main(argv):
    csvpath = argv[0]
    distance_metric = "cos" if len(argv)==1 else argv[1]
    odf = pd.read_csv(csvpath)
    df = odf
    majority_threshold = 0.5

    combs_arr = []

    filepath = csvpath.split("/")[-1]
    model_arr = filepath.split("_")[1].split("-")

    modelA = model_arr[0]
    modelB = model_arr[1]
    modelC = model_arr[2]
    final_fault = filepath.split("_")[-1].split(".")[0]
    if final_fault.startswith("err"):
        final_fault = filepath.split("_")[-2] + "_" + filepath.split("_")[-1].split(".")[0]

    dataset = filepath.split("_")[2]

    dataset_classes = {"cifar10": 10, "gtsrb": 43, "pneumonia": 2}
    num_classes = dataset_classes[dataset]

    ens_df = df
    ens_df["num_unique"] = ens_df[["predicted_A", "predicted_B", "predicted_C"]].nunique(axis=1)


    temp_df = ens_df[["predicted_A", "predicted_B", "predicted_C"]].mode(axis=1).iloc[:,0:1].astype(int)
    temp_df.rename(columns={ temp_df.columns[0]: "simple_maj" }, inplace = True)
    ens_df["simple_maj"] = temp_df["simple_maj"]



    total_samples = ens_df.shape[0]


    logit_root = "./logits/"
    logit_path = logit_root + dataset + "_" + final_fault + "_"

    logitA = np.loadtxt(logit_path + modelA)
    logitB = np.loadtxt(logit_path + modelB)
    logitC = np.loadtxt(logit_path + modelC)

    logitAvg = (logitA + logitB + logitC)/3
    uniform_avg_preds = np.argmax(logitAvg, axis = 1).flatten()[:total_samples]
    ens_df["uniform_avg"] = uniform_avg_preds


    static_weighted_voting(ens_df, dataset, total_samples, num_classes)

    remix_voting(ens_df, dataset, total_samples, num_classes, logitA, logitB, logitC, distance_metric)


    if dataset == "pneumonia":
        print("Best Individual BA: ", individual_model(ens_df, "f1"))
    else:
        print("Best Individual BA: ", individual_model(ens_df, "ba"))
 
    if dataset == "pneumonia":
        print("Uniform Majority F1: ", f1_score(ens_df["ground_truth"], ens_df["simple_maj"], average="binary"))
        print("Uniform Average F1: ", f1_score(ens_df["ground_truth"], ens_df["uniform_avg"], average="binary"))
        print("Static Weighted Majority F1: ", f1_score(ens_df["ground_truth"], ens_df["weighted_maj"], average="binary"))
        ens_df = ens_df[(ens_df["remix_avg"] >= majority_threshold)]
        print("Remix F1: ", f1_score(ens_df["ground_truth"], ens_df["remix"], average="binary"))
    else:
        print("Uniform Majority BA: ", balanced_accuracy_score(ens_df["ground_truth"], ens_df["simple_maj"]))
        print("Uniform Average BA: ", balanced_accuracy_score(ens_df["ground_truth"], ens_df["uniform_avg"]))
        print("Static Weighted Majority BA: ", balanced_accuracy_score(ens_df["ground_truth"], ens_df["weighted_maj"]))
        ens_df = ens_df[(ens_df["remix_avg"] >= majority_threshold)]
        print("Remix BA: ", balanced_accuracy_score(ens_df["ground_truth"], ens_df["remix"]))


if __name__ == "__main__":
    main(sys.argv[1:])

