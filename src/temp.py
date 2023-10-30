from tqdm import trange
from time import sleep

postfix = [0.]
epoch_bar_format = 'Training: {percentage:3.0f}%|{bar:25}| Epoch {n_fmt}/{total_fmt}, Loss={postfix[0]}'
batch_bar_format = '\tBatch {n_fmt}/{total_fmt}: {percentage:3.0f}%|{bar:25}|'
for i in trange(10, bar_format=epoch_bar_format, postfix=postfix, initial=5):
    for j in trange(5, leave=False, bar_format=batch_bar_format):
        sleep(0.2)
    postfix[0] = i * 0.1
