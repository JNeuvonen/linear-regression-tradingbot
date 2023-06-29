# What is it?
This is an automated trading system that integrates Binance and Glassnode datasets, implements data preprocessing and feature extraction, utilizes linear regression for model training, evaluates model performance across various parameters, and enables easy deployment in a live trading environment using Docker. The project is something that I hacked together in two weeks since I wanted to learn about ML & MLops, not anything I would pursue seriously.

Live test bot deployed using $25 balance: http://52.195.167.170/

# Features
1. Automated data scraping from Binance public data-library
2. Support for glassnode datasets by adding glassnode datasets to dir "glassnode_data" in the project root, and the dataset's desired features get automatically added to the final training dataset.
3. Pre-processes the given data: drops unwanted cols, combines datasets, creates desired features, creates train/valid split etc.
4. Automatically starts training the model with the desired features using linear regression.
5. Automatically compare epochs performance on an out-of-sample backtest with different strategy parameters and save the models that perform within the top 5% in the out-of-sample backtest.
6. Deploy the automated trading system easily in a live environment using Docker.
7. Easily customizable & trained for all cryptocurrencies that are traded on Binance
8. Easily customizable model features.

# Run

1. `cd src`
2. `python3 -m venv env`
3. activate env
    1. mac/linux: `source env/bin/activate`
    2. windows: `env\Scripts\activate`
      
4. `pip install -r requirements.txt`
5. `python main.py`
6. Run for desired number of epochs, you can test the models while it is training by using `python backtest.py {model_epoch_number}`, more details: `python backtest.py --help`

# Backtest/Validate models (dependencies must be installed)

1. `cd src`
2. `python backtest.py {model_epoch_number}`

More details: `python backtest.py --help`

