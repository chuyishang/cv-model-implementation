import pandas as pd
import numpy as np

PATH = 'output/resnet152/preds.npy'

data = np.load(PATH)
data = np.argmax(data, axis=1)

df = pd.DataFrame(data)
df.index += 1
df.to_csv('resnet152_18epoch3_submission.csv', header=['Category'], index_label='Id')