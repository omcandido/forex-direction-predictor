from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile
from typing import Union
from enum import IntEnum

import datetime

COL_NAMES = ['date', 'open', 'high', 'low', 'close', 'volume']

# Feature index class
class OHLC (IntEnum):
    OPEN   = 0
    HIGH   = 1
    LOW    = 2
    CLOSE  = 3

class TicksDataset(Dataset):
    """Ticks dataset."""

    def __init__(self, input_files, n_past, lookahead, tolerance, transform=None, target_transform=None):
        self.input_files = input_files
        self.n_past = n_past
        self.lookahead = lookahead
        self.tolerance = tolerance
        self.transform = transform
        self.target_transform = target_transform
        self.ticks, self.epochs = csv_to_ticks(input_files)
        self.valid_id = get_valid_ticks(self.epochs, n_past, lookahead, tolerance)
        
    def __len__(self):
        return len(self.valid_id)

    def __getitem__(self, idx):
        i = self.valid_id[idx]
        epoch = self.epochs[i].copy()
        candlesticks = self.ticks[i-self.n_past:i].copy()
        past_epochs = self.epochs[i-self.n_past:i].copy()
        x = {'candlesticks': candlesticks, 'past_epochs': past_epochs, 'epoch': epoch}
        if self.transform:
            x = self.transform(x)
        label = get_label(self.ticks, self.lookahead, i).copy()
        y = {'label': label}
        if self.target_transform:
            y = self.target_transform(y)
        return x['candlesticks'], y['label']

    
def extract_data(file, col_names = COL_NAMES) -> Union[pd.DataFrame, None]:
    with ZipFile(file, 'r') as zip:
        for f in zip.namelist():
            if '.csv' in f:
                df = pd.read_csv(zip.open(f), sep=';', names=col_names)
                zip.close()
                return df

def csv_to_ticks(
    input_files, 
    col_names = COL_NAMES
):
    data = []
    for file in tqdm(input_files):
        df = extract_data(file, col_names) if str(file).endswith('.zip') else pd.read_csv(file, sep=";", names=col_names)
        data.append(df)
    print("Sorting data...")
    data = pd.concat(data)
    data = data.drop_duplicates(subset=['date']) # For some reason, there might be duplicate ticks.
    data.reset_index(drop=True, inplace=True)
    # Convert date to epochs
    epochs = pd.to_datetime(data['date'], format='%Y%m%d %H%M%S')
    epochs = datetime_to_epoch(epochs).astype('int32').to_numpy().copy()
    # Extract the columns OPEN, HIGH, LOW, CLOSE as a numpy array
    ticks = data[['open', 'high', 'low', 'close']].astype('float32').to_numpy().copy()
    print("Done!")
    del data
    return ticks, epochs

def datetime_to_epoch(datetime):
    return (datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

def get_valid_ticks(
    epochs: np.ndarray,
    n_past: int,
    horizon: int,
    tolerance: float = 1.
):
    valid_id = []
    # Loop through all possible time windows of the given length in the dataset.
    # TODO: Tolerance might make you miss some valid windows at the beginning of the dataset.
    # The i_th index is the current tick.
    # ┌───────────── n_past=20 ─────────────┐ i ┌─── horizon=10 ──┐
    # □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □ ■ □ □ □ □ □ □ □ □ □ ▣
    # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9   0 1 2 3 4 5 6 7 8 9 
    windows = tqdm(range(n_past, len(epochs) - horizon))
    for i in windows:
        # Check that the current window meets the time tolerance.
        if (epochs[i] - epochs[i - n_past]) * tolerance > n_past * 60:
            continue
        
        # Check that the target tick is present.        
        if not (horizon * 60 + epochs[i]) in epochs[i:i + horizon+1]:
            continue
        
        valid_id.append(i)
    print("Samples identified: {} ({:.4%}).".format(len(valid_id), len(valid_id) / len(epochs)))
    return valid_id

def ticks_to_candlesticks(ticks:np.ndarray, width:int):
    n_past = len(ticks)
    dtype = ticks.dtype
    assert n_past % width == 0, "candle width must be divisor of past"
    n_candlesticks = int(n_past/width)
    candlesticks = []
    for i in range(n_candlesticks):
        window_i = ticks[i*width:(i+1)*width]
        candle = [window_i[0,  OHLC.OPEN],
                window_i[:,  OHLC.HIGH].max(),
                window_i[:,  OHLC.LOW].min(),
                window_i[-1, OHLC.CLOSE]]
        candlesticks.append(candle)
    candlesticks = np.array(candlesticks, dtype=dtype)
    return candlesticks


def get_label(ticks:np.ndarray, horizon:int, index:int):
    # Calculate LOW(horizon) - HIGH(present)
    y_high = ticks[index+horizon, OHLC.LOW] - ticks[index, OHLC.HIGH]
    # If >0, then the value went UP.
    y_high = y_high*(y_high>0)
    # Calculate HIGH(horizon) - LOW(present)
    y_low = ticks[index+horizon, OHLC.HIGH] - ticks[index, OHLC.LOW]
    # If <0, then the value went DOWN.
    y_low = y_low*(y_low<0)
    # Values are complementary, so we can safely add them to merge them.
    y = (y_high+y_low)
    return y
    
class ToCandlesticks(object):
    def __init__(self, width:int):
        self.width = width
    def __call__(self, x):
        candlesticks = ticks_to_candlesticks(x['candlesticks'], self.width)
        return {'candlesticks': candlesticks, 'epoch': x['epoch']}
    
class FlipAxes(object):
    def __call__(self, x):
        x['candlesticks'] = x['candlesticks'].transpose(1, 0)
        return x
    
class Threshold(object):
    def __init__(self, threshold:float):
        self.threshold = threshold
    def __call__(self, y):
        if  (np.abs(y['label']) < self.threshold):
            y['label'] = 0
        else:
            y['label'] = y['label']
        return y

class ToMulticlass(object):
    def __call__(self, y):
        if y['label'] > 0:
            y['label'] = 1
        elif y['label'] < 0:
            y['label'] = 2
        else:
            y['label'] = 0
        
        return y
    
class ToPipDifference(object):
   # Subtract the subsequent closing price from the last one.
    # This gives a stationary time series.
    def __init__(self, pip=1e-4) -> None:
        self.pip = pip
    def __call__(self, x):
        # Get all the open prices.
        open_prices = x['candlesticks'][:, OHLC.OPEN].reshape(-1, 1)
        # Roll everything to the left so it's easier to calculate the deltas.
        open_prices = np.roll(open_prices, -1, axis=0)
        # Approximate the current open price with the last close price.
        open_prices[-1] = x['candlesticks'][-1][OHLC.CLOSE]
        x['candlesticks'] = (x['candlesticks'] - open_prices) / self.pip
        return x
    
def scale(x, minimum=-50, maximum=50, clip=True):
    res = (x - minimum) / (maximum - minimum) 
    res = res*2-1
    if clip:
        res = res.clip(-1, 1)
    return res