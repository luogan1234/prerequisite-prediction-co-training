import os

if not os.path.exists('dataset/'):
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/prerequisite_dataset.zip')
    os.system('unzip prerequisite_dataset.zip')