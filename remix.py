# This is the main script for training ensembles and extracting feature matrices for Remix

import sys
import numpy as np
from diversity_utils import *
from distance import *

from numpy import dot
from numpy.linalg import norm
import pandas as pd

import matplotlib.pyplot as plt

import shap

from omnixai.data.image import Image
from omnixai.explainers.vision import CounterfactualExplainer, IntegratedGradientImage, SmoothGrad, LimeImage

import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import argparse



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



parser = argparse.ArgumentParser(description='Train model with fault params')
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'gtsrb', 'pneumonia'], default='mnist')
parser.add_argument('--final_fault', type=str, default="label_err-30")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--conf_threshold', type=float, default=0.7)
parser.add_argument('--samples', type=int, default=30)
parser.add_argument('--natural', action='store_true')
parser.add_argument('--step_size', type=int, default=1000)

parser.add_argument('--xai', type=str, choices=['shap', 'cfe', 'ig', 'sg', 'lime'], default='sg')

parser.add_argument('--modelA', type=str, default="ConvNet")
parser.add_argument('--modelB', type=str, default="DeconvNet")
parser.add_argument('--modelC', type=str, default="VGG11")

args = parser.parse_args()

dataset = args.dataset
final_fault = args.final_fault
num_epochs = args.epochs
batch_size = args.batch_size
step_size = args.step_size

modelA_name = args.modelA
modelB_name = args.modelB
modelC_name = args.modelC



def train_and_get_predictions(dataset, final_fault, model_name, x_train, y_train, x_test, y_test, num_epochs, batch_size):
    model = get_trained_model(dataset, model_name, x_train, y_train, num_epochs, batch_size)
    scores = model.evaluate(x_test,
                            y_test,
                            batch_size=batch_size,
                            verbose=0)
    print('Test accuracy for :', model_name, scores[1])

    predictednp = model.predict(x_test)
    predictions = np.argmax(predictednp, axis = 1).flatten()
    high_softmax = np.max(predictednp, axis = 1).flatten()

    np.savetxt("./logits/" + dataset + "_" + final_fault + "_" + model_name, predictednp, fmt='%f')

    return model, predictions, high_softmax


def squeeze(orig_heatmap, xai_mode):
    if xai_mode != "lime":
        if dataset != "pneumonia" and dataset != "mnist":
            heatmap = tf.image.rgb_to_grayscale(orig_heatmap).numpy()
        else:
            heatmap = orig_heatmap
        return np.squeeze(heatmap, axis=2)
    else:
        return orig_heatmap


def calc_corr(heatmapA, heatmapB):
    sim = (pearson_r(heatmapA, heatmapB) + pearson_r(heatmapB, heatmapA)) / 2
    return sim


def pearson_r(heatmapA, heatmapB):
    sim = np.corrcoef(heatmapA, heatmapB)
    return sim[0][1] * sim[0][1]


def calc_frob(heatmapA, heatmapB):
    return norm(heatmapA - heatmapB)


def calc_cos(heatmapA, heatmapB):
    a = heatmapA.flatten()
    b = heatmapB.flatten()
    return dot(a, b)/(norm(a)*norm(b))


def calculate_sparsity(heatmap, xai_mode):
    if xai_mode == "sg" or xai_mode == "ig":
        threshold = 1e-02
    else:
        threshold = 1e-05
    val_arr = np.absolute(heatmap)
    sparsity = 1.0 - ( (val_arr > threshold).sum() / float(val_arr.size) )
    return sparsity


def calc_diversity_columns(xai_mode, df, explainA, explainB, explainC):
    sparsity_arr1 = []
    sparsity_arr2 = []
    sparsity_arr3 = []

    corr_arr1 = []
    corr_arr2 = []
    corr_arr3 = []

    frob_arr1 = []
    frob_arr2 = []
    frob_arr3 = []

    cos_arr1 = []
    cos_arr2 = []
    cos_arr3 = []

    len_samples = explainA.shape[0]


    for i in range(len_samples):

        heatmapA = squeeze(explainA[i], xai_mode)
        heatmapB = squeeze(explainB[i], xai_mode)
        heatmapC = squeeze(explainC[i], xai_mode)

        sparsity_A = calculate_sparsity(heatmapA, xai_mode)
        sparsity_arr1.append(sparsity_A)

        sparsity_B = calculate_sparsity(heatmapB, xai_mode)
        sparsity_arr2.append(sparsity_B)

        sparsity_C = calculate_sparsity(heatmapC, xai_mode)
        sparsity_arr3.append(sparsity_C)

        corr = calc_corr(heatmapA, heatmapB)
        corr_arr1.append(corr)

        corr = calc_corr(heatmapB, heatmapC)
        corr_arr2.append(corr)

        corr = calc_corr(heatmapA, heatmapC)
        corr_arr3.append(corr)
        
        corr = calc_frob(heatmapA, heatmapB)
        frob_arr1.append(corr)

        corr = calc_frob(heatmapB, heatmapC)
        frob_arr2.append(corr)

        corr = calc_frob(heatmapA, heatmapC)
        frob_arr3.append(corr)
        
        corr = calc_cos(heatmapA, heatmapB)
        cos_arr1.append(corr)

        corr = calc_cos(heatmapB, heatmapC)
        cos_arr2.append(corr)

        corr = calc_cos(heatmapA, heatmapC)
        cos_arr3.append(corr)

    df["sparsity_A"] = sparsity_arr1
    df["sparsity_B"] = sparsity_arr2
    df["sparsity_C"] = sparsity_arr3

    df["corr_AB"] = corr_arr1
    df["corr_BC"] = corr_arr2
    df["corr_AC"] = corr_arr3
    df["frob_AB"] = frob_arr1
    df["frob_BC"] = frob_arr2
    df["frob_AC"] = frob_arr3
    df["cos_AB"] = cos_arr1
    df["cos_BC"] = cos_arr2
    df["cos_AC"] = cos_arr3


def calc_shap_values(model, background, x_test, predictions, indices):
    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(np.take(x_test, indices, 0))
    relevant_pred = [predictions[i] for i in indices]
    shap_values = [shap_values[pred][i] for i, pred in enumerate(relevant_pred)]
    shap_values = np.asarray(shap_values)
    return shap_values


def calc_lime_values(model, x_test, predictions, indices):
    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=5, max_dist=6, ratio=0.5)

    mask_list = []

    for index in indices:
        explanation = explainer.explain_instance(x_test[index], model.predict, segmentation_fn=segmenter)
        _, mask = explanation.get_image_and_mask(predictions[index], positive_only=True, hide_rest=True)
        mask_list.append(mask)

    mask_list = np.asarray(mask_list, dtype=np.float32)
    return mask_list


def calc_cfe_ig_sg_values(xai_mode, model, x_test, indices):
    x_test = Image(x_test.astype('float32'), batched=True)

    if dataset == "mnist" or dataset == "pneumonia":
        preprocess_func = lambda x: np.expand_dims(x.to_numpy(), axis=-1)
    else:
        preprocess_func = lambda x: x.to_numpy()

    if xai_mode == "ig":
        explainer = IntegratedGradientImage(
            model=model,
            preprocess_function=preprocess_func
        )
    elif xai_mode == "sg":
        explainer = SmoothGrad(
            model=model,
            preprocess_function=preprocess_func
        )
    else: # cfe
        explainer = CounterfactualExplainer(
            model=model,
            preprocess_function=preprocess_func
        )
    explanations = explainer.explain(x_test[indices])
    exp = explanations.get_explanations()
    return exp


def extract_explanations(batch_explain, xai_mode):
    explain_list = []
    for explanation in batch_explain:
        if xai_mode == "ig" or xai_mode == "sg":
            explain_list.append(explanation["scores"])
        else:
            explain_list.append(explanation["cf"])
    return np.asarray(explain_list)


def generate_csv(xai_mode, indices, y_test, predictions_A, predictions_B, predictions_C, high_softmax_A, high_softmax_B, high_softmax_C, explainA, explainB, explainC):
    list_name = ['index', 'ground_truth']
    df = pd.DataFrame(columns=list_name)


    df["index"] = indices
    df["ground_truth"] = np.take(y_test, indices).tolist()
    predicted_A = np.take(predictions_A, indices).tolist()
    df["predicted_A"] = predicted_A
    df["highest_softmax_A"] = np.take(high_softmax_A, indices).tolist()
    predicted_B = np.take(predictions_B, indices).tolist()
    df["predicted_B"] = predicted_B
    df["highest_softmax_B"] = np.take(high_softmax_B, indices).tolist()
    predicted_C = np.take(predictions_C, indices).tolist()
    df["predicted_C"] = predicted_C
    df["highest_softmax_C"] = np.take(high_softmax_C, indices).tolist()


    calc_diversity_columns(xai_mode, df, explainA, explainB, explainC)

    file_name = "./remix_results/" + xai_mode + "_" + "-".join([modelA_name,modelB_name,modelC_name]) + "_" + dataset + "_" +  final_fault + ".csv"
    df.to_csv(file_name)


def append_ex(explain_list, new_explain):
    if explain_list is not None:
        explain_list = np.concatenate((explain_list, new_explain))
    else:
        explain_list = new_explain
    return explain_list


def main(argv):
    conf_threshold = args.conf_threshold
    len_samples = args.samples

    if len_samples == 30:
        dataset_samples = {"cifar10": 10000, "gtsrb": 12630, "pneumonia": 624}
        len_samples = dataset_samples[dataset]

    xai_mode = args.xai

    symmetric = not args.natural

    (x_train, y_train), (x_test, y_test) = load_training_data(dataset, final_fault, symmetric)
    
    if xai_mode == "lime" and (dataset == "mnist" or dataset == "pneumonia"):
        x_train = np.repeat(x_train, 3, axis=3)
        x_test = np.repeat(x_test, 3, axis=3)

    modelA, predictions_A, high_softmax_A  = train_and_get_predictions(dataset, final_fault, modelA_name, x_train, y_train, x_test, y_test, num_epochs, batch_size)
    modelB, predictions_B, high_softmax_B  = train_and_get_predictions(dataset, final_fault, modelB_name, x_train, y_train, x_test, y_test, num_epochs, batch_size)
    modelC, predictions_C, high_softmax_C  = train_and_get_predictions(dataset, final_fault, modelC_name, x_train, y_train, x_test, y_test, num_epochs, batch_size)


    full_index = range(len_samples)

    start = 0
    stop = len_samples
    if dataset == "pneumonia":
        step = 100
    else:
        step = step_size
    range_samples = [range(n, min(n+step, stop)) for n in range(start, stop, step)]


    explainA_list = None
    explainB_list = None
    explainC_list = None

    for indices in range_samples:

        if xai_mode == "cfe" or xai_mode == "ig" or xai_mode == "sg":
            heatmap_values_A = calc_cfe_ig_sg_values(xai_mode, modelA, x_test, indices)
            explainA = extract_explanations(heatmap_values_A, xai_mode)
            heatmap_values_B = calc_cfe_ig_sg_values(xai_mode, modelB, x_test, indices)
            explainB = extract_explanations(heatmap_values_B, xai_mode)
            heatmap_values_C = calc_cfe_ig_sg_values(xai_mode, modelC, x_test, indices)
            explainC = extract_explanations(heatmap_values_C, xai_mode)

        elif xai_mode == "lime":
            explainA = calc_lime_values(modelA, x_test, predictions_A, indices)
            explainB = calc_lime_values(modelB, x_test, predictions_B, indices)
            explainC = calc_lime_values(modelC, x_test, predictions_C, indices)

        else: #shap
            background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
            explainA = calc_shap_values(modelA, background, x_test, predictions_A, indices)
            explainB = calc_shap_values(modelB, background, x_test, predictions_B, indices)
            explainC = calc_shap_values(modelC, background, x_test, predictions_C, indices)

        explainA_list = append_ex(explainA_list, explainA)
        explainB_list = append_ex(explainB_list, explainB)
        explainC_list = append_ex(explainC_list, explainC)


    generate_csv(xai_mode, full_index, y_test, predictions_A, predictions_B, predictions_C, high_softmax_A, high_softmax_B, high_softmax_C, explainA_list, explainB_list, explainC_list)


if __name__ == "__main__":
    main(sys.argv[1:])

