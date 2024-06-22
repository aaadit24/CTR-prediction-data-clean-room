# CTR Prediction in Data Clean Room

This project demonstrates a prototype for Click-Through Rate (CTR) prediction using a Data Clean Room approach. It includes data preprocessing, aggregate statistics calculation, machine learning model training, and synthetic data generation and evaluation.

## Setup

1. Clone this repository:

`git clone https://github.com/aaadit24/CTR-prediction-data-clean-room.git`
`cd ctr-prediction-data-clean-room`

2. Install the required packages:

`pip install -r requirements.txt`

## Usage

1. Place your input data files in the `decrypted_file/train/` directory:
- `train_data_feeds.csv`
- `train_data_ads.csv`

2. Run the Data Clean Room script:
`python data_clean_room.py`
This script will:
- Load and preprocess the data
- Calculate aggregate statistics
- Generate synthetic data
- Evaluate the synthetic data

3. To train and evaluate the CTR prediction model:
`python ctr_prediction.py`
This script will:
- Create and train the CTR prediction model
- Evaluate the model's performance
- Generate a feature importance plot

## Output

The scripts will generate several output files:
- `distribution_comparison_*.png`: Plots comparing the distributions of original and synthetic data for each feature
- `feature_importance.png`: A bar plot showing the importance of each feature in the CTR prediction model

## Approach

This prototype demonstrates a Data Clean Room approach for CTR prediction:

1. Data Integration: Merges publisher (feeds) and advertiser (ads) data securely.
2. Aggregate Statistics: Calculates key statistics without exposing individual records.
3. Machine Learning: Trains a Random Forest model for CTR prediction.
4. Synthetic Data: Generates and evaluates synthetic data to preserve privacy.
