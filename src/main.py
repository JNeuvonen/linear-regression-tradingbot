from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import requests
import os
import zipfile
import shutil
import subprocess
import pandas as pd


class ScrapeJob:
    def __init__(self, url, target_dataset, columns_to_drop):
        self.url = url
        self.market_type = None
        self.klines = None
        self.candle_interval = None
        self.dataseries_interval = None
        self.pair = None
        self.margin_type = None
        self.prefix = None
        self.columns_to_drop = columns_to_drop
        self.target_dataset = target_dataset
        self.path = self.process_url()

    def is_spot_url(self):
        return "spot" in self.url

    def get_prefix(self):
        return self.pair + "_" + self.market_type + \
            "_" + self.candle_interval + "_"

    def process_url(self):
        if self.is_spot_url():
            url_parts = self.url.split("/")
            self.dataseries_interval = url_parts[5]
            self.klines = url_parts[6]
            self.pair = url_parts[7]
            self.candle_interval = url_parts[8]
            self.market_type = "spot"
            return os.path.join(
                "..", 'data', self.market_type, self.klines, self.pair, self.candle_interval)
        else:
            url_parts = self.url.split("/")
            prefix_index = url_parts.index('?prefix=data') + 1
            self.margin_type = url_parts[prefix_index+1]
            self.dataseries_interval = url_parts[prefix_index+2]
            self.klines = url_parts[prefix_index+3]
            self.pair = url_parts[prefix_index+4]
            self.candle_interval = url_parts[prefix_index+5]
            type_of_endpoint = endpoint_type(self.url)
            self.candle_interval = get_datatick_interval(
                type_of_endpoint, self.candle_interval)
            self.market_type = "futures"
            return os.path.join("..", 'data', self.market_type, self.margin_type,
                                self.klines, self.pair, self.candle_interval)


def is_spot_url(url):
    return "spot" in url


def endpoint_type(url):
    return "metrics" in url


def get_datatick_interval(scrape_type, fallback):
    if scrape_type == "metrics":
        return "5m"

    return fallback


def scrape_job(scrape_object):

    service = Service()
    driver = webdriver.Chrome(service=service)
    driver.get(scrape_object.url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    trs = soup.find_all('tr')
    count = 0
    for tr in trs:
        count += 1
        if count < 3:
            continue
        td = tr.find('td')
        anchor = td.find('a')
        href = anchor['href']

        if href.endswith('.zip') and "CHECKSUM" not in href:
            download_file(href)

    driver.quit()

    unzip_files('scraped_data', scrape_object)


def download_file(url):
    directory = 'scraped_data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    local_filename = url.split('/')[-1]
    local_path = os.path.join(directory, local_filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path


def unzip_files(input_dir, scrape_object):

    if not os.path.exists(scrape_object.path):
        os.makedirs(scrape_object.path)

    files = os.listdir(input_dir)

    for file in files:
        if file.endswith('.zip'):
            file_path = os.path.join(input_dir, file)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(scrape_object.path)
    subprocess.call(["python", "combine_scraped_data.py"] +
                    [scrape_object.path.replace("\\", "/")])

    shutil.rmtree(input_dir)


def convert_url_to_data_path(url):
    url_parts = url.split("/")
    dataseries_interval = url_parts[5]
    klines = url_parts[6]
    pair = url_parts[7]
    candle_interval = url_parts[8]

    return os.path.join("..", "..", 'data_v2', "spot", klines, pair, candle_interval)


def pre_process(scrape_objects, strat_name, glassnode_data=[]):
    target_prefix = ""
    for scrape_object in scrape_objects:

        subprocess.call(["python", "concatenate_pair_datasets.py"] +
                        [scrape_object.path.replace("\\", "/"), scrape_object.pair, scrape_object.market_type, scrape_object.candle_interval, strat_name])

        if scrape_object.target_dataset:
            target_prefix = scrape_object.get_prefix()

    for scrape_object in scrape_objects:
        if len(scrape_object.columns_to_drop) > 0:
            csv_file = os.path.join(
                scrape_object.path, strat_name, scrape_object.get_prefix() + "combined_data.csv")
            cols_to_drop_w_prefix = []
            for item in scrape_object.columns_to_drop:
                cols_to_drop_w_prefix.append(
                    scrape_object.get_prefix().lower() + item)
            df = pd.read_csv(csv_file.replace("\\", "/"))
            cols_to_drop_w_prefix = [
                col for col in cols_to_drop_w_prefix if col in df.columns]
            df.drop(cols_to_drop_w_prefix, axis=1, inplace=True)
            df.to_csv(csv_file, index=False)

    paths = " ".join([scrape_object.path.replace("\\", "/") +
                     "/" + strat_name + "/" + scrape_object.get_prefix() + "combined_data.csv" for scrape_object in scrape_objects])

    glassnode_paths = " ".join(glassnode_data)

    subprocess.call(["python", "pre_process.py"] +
                    [paths, target_prefix, strat_name, glassnode_paths])


def train():
    subprocess.call(["python", "train.py"])


def get_glassnode_data_paths():
    ret = []
    for filename in os.listdir("../glassnode_data/"):
        ret.append("../glassnode_data/" + filename)

    return ret


def scrape(jobs):
    for job in jobs:
        scrape_job(job)


if __name__ == "__main__":
    LIST_OF_SCRAPE_JOBS = [
        ScrapeJob(
            "https://data.binance.vision/?prefix=data/spot/monthly/klines/IOTAUSDT/1h/",
            True,
            [],
        ),
        ScrapeJob(
            "https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1h/",
            False,
            ["open_price", "high_price", "low_price", "close_price",
                "volume", "quote_asset_volume", "number_of_trades", "ignore"],
        ),
    ]

    scrape(LIST_OF_SCRAPE_JOBS)
    glassnode_data = get_glassnode_data_paths()
    pre_process(LIST_OF_SCRAPE_JOBS, "linear_regr_v2",
                glassnode_data=glassnode_data)
    train()
