# **Tumor Detection from Breast Histopathology Images**

A Convolutional Neural Network (CNN) for detecting breast cancer from histopathology slides

📌 **Project Overview**

This project implements a deep-learning pipeline for detecting the presence of breast tumors using histopathology image patches. A Convolutional Neural Network (CNN) was trained using the Breast Histopathology Images dataset from Kaggle:

🔗 **Dataset link**: https://www.kaggle.com/datasets/alaminbhuyan/breast-histopathology-images

The final model is saved as Tumor_detector.keras, and can be used directly for inference (prediction) without retraining.

The entire source code is located inside the Code/ directory.

**📁 Repository Structure**

```bash 
Tumor-Detection/
│
├── Code/                     # All scripts for model training, inference & preprocessing
│   ├── model.py              # Defines CNN architecture and creates the model
│   ├── train.py              # Trains the model and saves model + history
│   ├── inference.py          # Loads trained model and predicts tumor presence
│   ├── preprocessing.py      # Splits dataset into train/test/validate
│   ├── plots.py              # Generates accuracy & loss plots from training history
│   └── ...                  
│
├── model/                    # Saved trained model
│   └── Tumor_detector.keras
│
├── data/                     # Final processed dataset (train/test/validate)
│   ├── train/
│   ├── test/
│   └── validate/
│
├── Original_dataset/         # Raw dataset downloaded from Kaggle
│
├── plots/                    # Generated accuracy and loss curves
│
├── Notebooks/                # Jupyter notebook with raw/experimental code
│
├── requirements.txt
└── README.md
```

**🧪 Dataset Preparation (Preprocessing)**

Before training, you must prepare your dataset.

**1.** Download the dataset from Kaggle.

**2.** Create the directory:

```swift
Tumor-Detection/Original_dataset/
```

**3.** Place the extracted Kaggle files exactly as follows:

```markdown
Original_dataset/
└── IDC_regular_ps50_idx5/
    ├── negative_IDC/
    └── positive_IDC/
```

4. Run the preprocessing script:

```bash
cd Code
python preprocessing.py
```

This will:

- Create train/, test/, and validate/ datasets.

- Balance dataset sizes according to your rules.

- Store all output inside the top-level /data/ directory.


**Model Definition (model.py)**

- Contains the architecture of the CNN.

- Must be imported where the training will be done:

```python
from model import define_model
model = define_model()
```

**Training the Model (train.py)**

This script:

- Loads the model from model.py

- Trains it on the processed dataset

Saves:

- Tumor_detector.keras inside the /model/ directory

- history.pkl containing training history

**Important:** train.py must be run from the same directory as model.py

*To train:*

```bash
cd Code
python train.py
```

**Plotting Training Curves (plots.py)**

- Uses the saved history.pkl file from training.

- Generates accuracy.png and loss.png

- Saves them automatically into:
 ```bash
 ../plots/
```
To generate plots:

```bash
cd Code
python plots.py
```
**🔍 Running Inference (Prediction)**

The inference script loads the trained model and predicts whether a tumor is present in a single image.

**Usage:**
```bash
cd Code
python inference.py path/to/image.png
```
**Output Interpretation:**

- **"Tumorous"** → Tumor present

- **"Negative"** → Tumor absent

Example:

```bash
python inference.py ../example_images/sample.png
```

**For Users Who ONLY Want to Predict**

They only need:

- Code/inference.py

- model/Tumor_detector.keras

- requirements.txt

Steps:

```bash
git clone https://github.com/AnovuyoNotsonono/Machine-learning.git
cd Tumor-Detection
pip install -r requirements.txt
python Code/inference.py path/to/image
```
No training required.

**🔁 For Users Who Want to Retrain the Model**

Steps:

**1.** Download dataset(recommended:from Kaggle)

**2.** Place inside /Original_dataset/ as described
 
**3.** Run:

```bash
python Code/preprocessing.py
python Code/train.py
python Code/plots.py
```
**4.** Use the updated model for inference.

**Installation**

Install dependencies:

```nginx
pip install -r requirements.txt
```
(You may use a Conda environment if preferred.)

**🧾 Requirements**

All required libraries are listed in requirements.txt.
This project was developed inside a Conda environment, but standard pip installation works as well.

**📓 Notebooks**

The Notebooks/ directory contains an exploratory Jupyter notebook with raw, uncleaned code used during early development. It can be used for quick testing or experimentation.


**Contact**

For questions or contributions, feel free to reach out or open an issue.
