# Features
1. Automated data scraping from Binance public data-library
2. Support for glassnode datasets by adding glassnode datasets to dir "glassnode_data" in the project root, and the dataset's desired features get automatically added to the final training dataset.
3. Pre-processes the given data: drops unwanted cols, combines datasets, creates desired features, creates train/valid split etc.
4. Automatically begin training the model with the desired features using linear regression.
5. Automatically compare epochs performance on an out-of-sample backtest with different strategy parameters and save the top 5% of performing models.
6. Deploy the automated trading system easily in a live environment using Docker.

