import os
import sys
import pickle
import random
import numpy as np

need = 10000
base_class_path = '/home/zhaozhiyuan/mmfewshot-main/base_statistics/2022-1-23-20:00'
classes = os.listdir(base_class_path)
print(classes)
base_statistics = {}

for i,c in enumerate(classes):
    
    if '.' in c:
        continue
    feats = os.listdir(os.path.join(base_class_path, c))
    need = min(5000, len(feats))
    feats = random.sample(feats, need)
    
    embeds = []
    count = 0
    # all features
    for feat in feats:
        try:
            f = open(os.path.join(base_class_path, c, feat), 'rb')
        except OSError:
            continue
        count += 1
        embed = pickle.load(f)
        embeds.append(embed)
        f.close()
    
    mean = np.mean(embeds, axis=0)
    cov = np.cov(np.array(embeds).T)
    base_statistics[c] = {}
    base_statistics[c]['mean'] = mean
    base_statistics[c]['variance'] = cov
    base_statistics[c]['N'] = len(embeds)
    
    print('{}/{}:{}'.format(i,c,count))
    
f = open(os.path.join(base_class_path, 'base_statistics.pt'), 'wb')
pickle.dump(base_statistics, f)
f.close()
