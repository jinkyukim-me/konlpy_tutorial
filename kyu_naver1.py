import pandas as pd
train_df = pd.read_csv('ratings_train.txt', sep='\t')

import re
train_df = train_df.fillna(' ')
