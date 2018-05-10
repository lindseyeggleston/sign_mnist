import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_dataset(filepath):
    '''Takes data from file and converts into numpy arrays'''
    img_df = pd.read_csv(filepath)
    y = img_df['label'].values
    X = img_df.drop('label', axis=1).values / 255
    return X, y


def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    
