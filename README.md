# ReMlX
Resilience for ML Ensembles using XAI - Remix for short

This repository is organized into 3 activities, in increasing order of time requirement.
Suggested times listed under each activity should only be used as a guide.


## Directory Layout

```
.
└── analyze_remix.py        # Script to run Remix on batch results and compare with ensembling baselines
├── arch                    # Folder containing TensorFlow implementations of model architectures
├── boost.py                # Script to invoke the boosting baseline
└── confFiles               # Contains pre-defined YAML files for experimental configs
└── data                    # Folder to store datasets
│   └── GTSRB               # Directory for the GTSRB dataset
├── diversity_utils.py      # Helper library for Remix (**not to be directly invoked**)
├── Dockerfile              # Used to generate Docker image for Remix
└── faulty_dataset          # Folder to store fault injected datasets
│   └── gtsrb
└── faulty_labels           # Folder to store Cleanlab generated dataset issues log files
│   └── issue_results_gtsrb.csv
├── logits                  # Folder to store logits obtained at inference
├── noise_matrix            # Folder to store noise transition matrices
├── noise_transition.py     # Script to generate noise transition matrix from Cleanlab log file
├── remix.py                # Main script to train and invoke Remix on an entire dataset
├── remix_example.py        # Script to train and then run Remix on a single test input and interactively see XAI generated feature matrices
├── requirements.txt        # List of dependencies
├── setup.sh                # Script to populate required empty folders for Remix
├── stack.py                # Script to invoke the dynamically weighted ensemble baseline (D-W Maj)
└── output                  # Output folder for Remix
└── TFDM                    # Folder for training data fault injector tool
```


## Installation

Requirements:

1. Python 3.8+
2. Pip 24+ (must be compatible with the Python version)
3. GPU (Note: Training and inference times can differ with GPU (>= 8 GB VRAM (Preferred: 16 GB VRAM)) capability, and VRAM availability).
4. Free disk space of about 2 GB.


### (Option A) Local Installation on a Machine with a NVIDIA GPU
**Note: This is the preferred and easiest option.**

1. Use pip to install the required dependencies.
```
pip install -r requirements.txt
```

2. Set the $PYTHONPATH environment variable so that it points to the TFDM fault injector.
This command must be executed at the start of each commamd terminal session.
Alternatively, it can added once to the `.bashrc` file.
```
export PYTHONPATH=$PYTHONPATH:[/path/to/Remix/TFDM]
```

For example, if Remix is located under /home, then simply enter the following line in the terminal (for each new session) or add it once to your `.bashrc` file.
```
export PYTHONPATH=$PYTHONPATH:/home/Remix/TFDM
```

### (Option B) Using Docker Image on Ubuntu 20.04 with NVIDIA CUDA

Assuming you have Docker (19.03+) installed, you will also need to enable Docker to access your GPU.
You will also need to ensure that you have the necessary permissions on the host to enable GUI forwarding.

1. To run the Docker image with GPU support, you will need to install `nvidia-container-runtime`.

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

2. Modify the Dockerfile so that it matches the CUDA version (default is 11.8) installed on your host machine.
If the CUDA version does not match, you will not be able to access the GPU for training (only CPU).

```
sudo docker build -t remix:latest .
export DISPLAY="127.0.0.1:10.0"
xhost +
sudo docker run -t -d -e DISPLAY=$DISPLAY -e DISPLAY_WIDTH=3840 -e DISPLAY_HEIGHT=2160 -v /tmp/.X11-unix:/tmp/.X11-unix --runtime=nvidia --gpus all --name remix_container remix
sudo docker exec -it remix_container /bin/bash
```

3. After running the desired activites below, run these commands to exit the Docker container.
```
exit
xhost -
```

### Datasets

1. GTSRB, a preprocessed version of the original, is included in this repository. No further preprocessing is required.
2. CIFAR-10 will be automatically downloaded if the dataset option is set to "cifar10" in Remix for the first time. No further preprocessing required.
3. Pneumonia can be downloaded [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Images must be resized to 128 x 128 pixels before use.
4. CIFAR-10 (128x128 resized) can be downloaded [here](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-128x128-resized-via-cai-super-resolution).


## Activity 1: Getting Started with a Minimum Working Example (No GPU required)
*Expected Activity Time: 2 min*

This minimum working example is designed so that users can see Remix functioning, without training the models from scratch.
We show an example of Remix on an ensemble of three models trained with GTSRB injected with 30% random asymmetric mislabelling.
We provide the results from pretrained models, averaged over 20 runs.

1. You should see two experimental result files under the `remix_results` folder. One is for GTSRB at 30% mislabelling, and one for 10%.

2. Pass the result to the `analyze_remix.py` script to calculate the balanced accuracy under Remix and compare it with other ensembling baseline approaches.
```
python analyze_remix.py remix_results/sg_ConvNet-DeconvNet-VGG11_gtsrb_label_err-30.csv
```
In most cases, you should observe that `Remix BA` listed on the last line of the output is the highest among baselines.
This result should be close to the values observed in **Figure 8a** in the paper.

Example Output:
```
Best Individual BA:  0.7115001947765695
Uniform Majority BA:  0.7515805114071039
Uniform Average BA:  0.7497890444761022
Static Weighted Majority BA:  0.7595186336339016
Remix BA:  0.8133484229313296
```

3. You can repeat this for GTSRB at 10% mislabelling by running this command.
While you should observe higher resilience overall due to the reduced fault amount, Remix is still expected to outperform baselines.
```
python analyze_remix.py remix_results/sg_ConvNet-DeconvNet-VGG11_gtsrb_label_err-10.csv
```


## Activity 2: Interactive Visual Example (GPU required)
*Expected Activity Time: 5 min (15 minutes with CPU)*

We provide an example of how to use Remix interactively, and visualize the XAI generated feature matrices.
You will need to have GUI access for this activity.
While we normally run Remix without any GUI, the purpose of this interactive demo is show what Remix is doing underneath the hood, for a single test input.


1. Run the following command. This will train the ensemble for GTSRB at 30% mislabelling, and run Remix-guided inference on a single test input.
```
python remix_example.py --dataset gtsrb --xai sg --final_fault label_err-30 --natural --modelA ConvNet --modelB DeconvNet --modelC VGG11
```

> Troubleshooting Tip: If you encounter an OOM error (Out of Memory) due to insufficient VRAM on your GPU, consider reducing the step_size (default is 1000).
> ```
> python remix_example.py --dataset gtsrb --xai sg --final_fault label_err-30 --natural --modelA ConvNet --modelB DeconvNet --modelC VGG11 --step_size 500
> ```

#### Explanation
You should see 4 figures open.
In the first 3 figures, you should see two subplots.
The number on the left indicates the predicted class by the individual model in the ensemble.
The score subplot on the right shows the visual heatmap representation of the generated Smooth Gradient explanation.
In the last figure (Figure 4), you should see a pop up of the original test image.

The main idea is that the models producing a similar looking feature explanation should have their predictions weighted lower than the model producing a more diverse feature explanation.
You will see a results output that shows the calculated weights and predicted class for each individual model in the ensemble, followed by the final predicted class by Remix.
Below is one example of the expected output format.
```
============================== Results ==============================

Weight for  ConvNet :     1.4078836     Prediction:  16
Weight for  DeconvNet :	  1.0421666     Prediction:  16
Weight for  VGG11 :	      0.8966636     Prediction:  5

Remix Prediction:  16
```
The GTSRB dataset has a total of 43 classes, where each predicted class number represents a road sign.
You may refer to [this image](https://miro.medium.com/v2/resize:fit:720/format:webp/1*IKGyG133iumZnyPPx7-Q0w.png) to see what each label class represents.


## Activity 3: Evaluating the Resilience of Remix Ensembles against Training Data Faults (GPU required)
*Expected Activity Time: 15 min (60 min with CPU)*

We provide an example of how to use Remix on the provided GTSRB dataset.
Our objective is to evaluate the resilience of a dynamically weighted ensemble on GTSRB against 30% mislabelling.

(1) First, we perform a new asymmetric fault injection.
(2) Then, we train each of the three models from scratch on the fault injected dataset.
(3) Next, we evaluate each model against the test dataset, save their logits and compute the feature diversity between models.
(4) Finally, Remix is applied, and its resilience is compared with that of ensembling baselines.

1. Ensure that the generated output folders are emptied by running the clean command.
**Notice:** *Once clean is initiated, you cannot return back to the minimum working example shown above, without pulling this repository again.*
```
./clean.sh
```

2. Run the following command.
```
python remix.py --dataset gtsrb --xai sg --final_fault label_err-30 --natural --modelA ConvNet --modelB DeconvNet --modelC VGG11
```
> Troubleshooting Tip: If you encounter an OOM error (Out of Memory) due to insufficient VRAM on your GPU, consider reducing the step_size (default is 1000).
> ```
> python remix.py --dataset gtsrb --xai sg --final_fault label_err-30 --natural --modelA ConvNet --modelB DeconvNet --modelC VGG11 --step_size 500
> ```

If you run this for the first time, you should see something like this: 
```
Initial Noise:  0.03652222703971027
[[  55    9    0 ...    0    0    0]
 [ 127 1746  100 ...    0    0    0]
 [   0  173 1275 ...    0    0    0]
 ...
 [   0   18    0 ...    0    0    0]
 [   0    0    0 ...    0   22   28]
 [   0    0    0 ...    0    0   57]]
Final Fault Rate:  0.3
New asymmetric fault injection performed
```
This means that a new asymmetric fault injection campaign has begun on the GTSRB dataset.
In this case, the background noise is detected as 3.65% and the fault amount in the dataset has been raised to 30% via fault injection.
The matrix printed out show a preview of the post-fault-injection noise transition matrix.

3. You will see a single file generated under the `remix_results` folder.

4. Pass the result to the `analyze_remix.py` script to calculate the balanced accuracy under Remix and compare it with other ensembling baseline approaches.
```
python analyze_remix.py remix_results/sg_ConvNet-DeconvNet-VGG11_gtsrb_label_err-30.csv
```
In most cases, you should observe that `Remix BA` listed on the last line of the output is the highest among baselines.
Of course, you must repeat this activity many times to make general observations.

Example Output:
```
Best Individual BA:  0.7115001947765695
Uniform Majority BA:  0.7515805114071039
Uniform Average BA:  0.7497890444761022
Static Weighted Majority BA:  0.7595186336339016
Remix BA:  0.8133484229313296
```


## License

Remix is released under an Apache 2.0 License.

