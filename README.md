# CSL7770 Assignment 1 - Shubh Goyal (B21CS073)

The repository contains the code and report for the first assignment of the course CSL7770 - Speech Understanding.

**Directory Structure**
```
.
├── question1_report.pdf
├── question2_report.pdf
├── question2
│   ├── TaskA
│   └── TaskB
└── README.md
```

**How to use this repository**

First, clone the repository to your local machine and navigate to the repository using the following commands:
```bash
git clone https://github.com/Shubh-Goyal-07/CSL7770_Assgn1.git
cd CSL7770_Assgn1
```

## Question 1

The report for Question 1 is present in the `question1_report.pdf` file

## Question 2

The report for Question 2 is present in the `question2_report.pdf` file

### Task A

Please follow the below instructions to run the code for Task A:

#### Instructions to replicate the results

1. **Navigate to the Task A directory**
   ```bash
   cd question2/TaskA
   ```

   This directory contains all the code files required to run Task A.
   The structure of the directory is as follows:
   ```
    .
    ├── logs
    ├── models
    ├── train_data
    ├── utils
    ├── processing.py
    ├── siamese.py
    ├── train.py
    ├── requirements.txt
    └── plots.ipynb
    ```

   The directory named `logs` currently contains the log files generated during the execution of the code. 
   If you re-run the code, the log files will be overwritten.

   The directory named `models` currently contains the trained models.
   If you re-run the training code, the models will be overwritten.

   The directory named `train_data` contains the embeddings (extracted features from spectrograms).

   The directory named `utils` contains the utility functions used in the code.

   The `plots.ipynb` file contains the code to generate the plots for the report.

2. **Download and extract the dataset**
   ```bash
   wget https://goo.gl/8hY5ER
   tar -zxvf 8hY5ER
   ```

   You must see a folder named `UrbanSound8K` in the current directory after running the above commands.

3. **Prepare the environment**
   ```bash
   conda create -n csl7770 -y
   conda activate csl7770
   conda install --file requirements.txt -y -c conda-forge -c nvidia -c pytorch
   ```
   
   The above command will create a new conda environment named `csl7770` and install all the required packages in it. It will take a few minutes to complete.

4. **Generate spectrograms**
   
   Run the file `processing.py` to generate the spectrograms for the audio files. The generated spectrograms will be saved in the `train_data/spectrograms/{window_name}` directory.
   
   This will also generate a log file named `processing_{window_name}.log` in the `logs` directory.
   
   ```bash
   python processing.py --window window_name --n window_size --overlap overlap_size
   ```

   The window_name can be one of the following: `hann`, `hamming`, `rectangular`.
   To generate spectrogram for all windows at once, keep the window_name as `all`.

   The code will also assess the window correctness using rmse and the results would be visible in the log file.

5. **Extracting features from spectrograms**
   
   To extract features from the generated spectrograms so as to train neural network and classical machine learning models, run the command given below.

   This will train a `resnet18` backbone siamese network to extract features from the spectrograms. The extracted features will be saved in the `train_data/embeddings/{window_name}_embeddings.csv` file.
   
   ```bash
   python siamese.py --window window_name --epochs num_epochs
   ```
 
   The window_name can be one of the following: `hann`, `hamming`, `rectangular`. To generate features for all windows at once, keep the window_name as `all`.

   The log file generated will be named `siamese_{window_name}.log`. Please use it to check the training progress.
    
   The trained siamese model will be saved as `models/siamese/model_{window_name}.pth` file.

6. **Training classifiers**
   
   To train classifiers on the generated data, run the command given below.

   ```bash
   python train.py --window window_name --model model_name --epochs num_epochs
   ```
   
   The window_name can be one of the following: `hann`, `hamming`, `rectangular`. 

   The models can be one of the following: `svm`, `rf`, `knn`, `dt`, `nn`, `cnn`.

   The log file generated will be named `train_{model_name}_{window_name}.log`. Please use it to check the training progress. 

   The trained model will be saved as `models/{model_name}/{window_name}.pth/pkl` file.


   **Note:** 
    - The argument `--epochs` is useful only when using `nn` or `cnn` models. It will be ignored for other models.
    - To train the `cnn` model, the spectrograms must be generated first using the `processing.py` script. The `cnn` model will use the spectrograms as training data.
    - The `nn`, `svm`, `rf`, `knn`, `dt` models will use the extracted features from the siamese network as training data. So, the siamese network must be trained first using the `siamese.py` script.
    - `svm` - Support Vector Machine, `rf` - Random Forest, `knn` - K-Nearest Neighbors, `dt` - Decision Tree, `nn` - Neural Network, `cnn` - Convolutional Neural Network.
  

  
### Task B

The code for task B is present in the `TaskB` directory.

```bash
cd question2/TaskB
```

The directory structure is as follows:
```
.
├── songs_data
|   ├── ghoomar_indian_rajasthani_folk.mp3
|   ├── love_the_way_u_lie_eminem_rap.mp3
|   ├── perfect_ed_pop.mp3
|   ├── sapna_jahan_bollywood.mp3
├── task_b.ipynb
```

The `songs_data` directory contains the songs used for the task.

The `task_b.ipynb` file contains the code for the task. It can be executed using the same environment created in Task A.
It contains the spectrograms of the four chosen songs.

The songs used are:
1. Ghoomar - Indian Folk Song
2. Love The Way You Lie - Rap Song
3. Perfect - Pop Song
4. Sapna Jahan - Bollywood Song