import os

if not os.path.exists('dataset/'):
    os.system('wget http://lfs.aminer.cn/misc/moocdata/data/prereq_co_training_dataset.zip')
    os.system('unzip prereq_co_training_dataset.zip')