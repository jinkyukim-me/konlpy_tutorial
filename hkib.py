import os
impoer re
from sklearn import datasets, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from konlpy.tag import Hannanum
from konlpy.tag import Kkma

import pandas as pd
import numpy as np

# 데이터 정리
dir_prefix = '../data'
target_dir = 'HKIB-20000'
cat_dirs = ['health', 'economy', 'science', 'education', 'culture', 'society','industry','leisure','politics']
cat_prefixes = ['건강','경제','과학','교육','문화','사회','산업','여가','정치']
files = os.listdir(dir_prefix + target_dir)

# 5분할된 텍스트 파일을 각각 처리
for file in files:
