import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('/content/drive/MyDrive/1730285881-Airbnb_Open_Data(in) (3) (1).csv')

from google.colab import drive
drive.mount('/content/drive')

df.head()

df.info()