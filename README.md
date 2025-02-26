# TEANet
## Data Preparation

The datasets used in this project are provided by the authors of TransUnet. You can access the processed data through the following link:

- **Synapse/BTCV**: [Get processed data here](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)

## Environment Setup

1. First, create an environment with Python 3.7.

   You can create a new virtual environment using `conda` or `venv`:

   - Using **conda**:
     ```bash
     conda create -n myenv python=3.7
     conda activate myenv
     ```

   - Using **venv**:
     ```bash
     python3.8 -m venv myenv
     source myenv/bin/activate  # On Linux/Mac
     myenv\Scripts\activate     # On Windows
     ```

2. After setting up the environment, install the required dependencies by running:
   ```bash
   pip install -r requirements.txt

## Train/Test

### Training

To train the model on the Synapse dataset, you can run the Python training script with the following command:

```bash
python train.py --dataset Synapse --root_path your DATA_DIR --max_epochs 200 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.001 --batch_size 16


