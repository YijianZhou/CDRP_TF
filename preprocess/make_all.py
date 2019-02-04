import os
os.system("python mk_dataset.py --sample_class 'events' --dataset_class 'train' &")
os.system("python mk_dataset.py --sample_class 'events' --dataset_class 'valid' &")
os.system("python mk_dataset.py --sample_class 'noise'  --dataset_class 'train' &")
os.system("python mk_dataset.py --sample_class 'noise'  --dataset_class 'valid' &")
os.system("python mk_dataset.py --sample_class 'ppk'    --dataset_class 'train' &")
os.system("python mk_dataset.py --sample_class 'ppk'    --dataset_class 'valid' &")

