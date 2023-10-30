import torch
import numpy as np
import cv2 as cv
import os.path as path
import sys

from model import CRNN
import config
from processing import WordProcessor, PositionMode, LineProcessor
from utils import ModelHandler

############ Required functions ############

def make_initial_setup():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        data_loader_args = {'num_workers': 1, 'pin_memory': True}
    else:
        device = torch.device('cpu')
        data_loader_args = {}

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return device, data_loader_args

def convert_to_word(word_embedding: torch.Tensor) -> str:
    word = ''
    for val in word_embedding:
        word += config.INDEXES_TO_TERMINALS[val]
    return word


def best_path(symbols_probability: torch.Tensor) -> torch.Tensor:
    """
    Implements best path decoding method to the probabilities matrix.

    Keyword arguments:
    symbols_probability: matrix of size CxT, where C - capacity of multiple terminals,
                         T - number of time-staps
                
    Return value:
    A recognized word coded with the config.MapTable
    """
    most_probable_symbols = torch.argmax(symbols_probability, dim=0)
    most_probable_label = torch.unique_consecutive(most_probable_symbols)
    mask = most_probable_label != 0
    most_probable_label[:mask.sum()] = most_probable_label[mask]
    most_probable_label[mask.sum():] = 0
    return most_probable_label


################### Main ###################

if __name__=='__main__':
    device, data_loader_args = make_initial_setup()

    model = CRNN(lstm_hidden_size=512,
                 lstm_num_layers=2,
                 mlp_hidden_size=2048,
                 dropout_p=0.3,
                 output_size=config.TERMINALS_NUMBER+1)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.LEARNING_RATE)
    handler = ModelHandler(model,
                           optimizer,
                           None,
                           params_file=path.join(config.PARAMS, config.BEST_PARAMS),
                           cur_params_file=path.join(config.PARAMS, config.CUR_PARAMS),
                           device=device)
    handler.recover()

    image = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

    line_processor = LineProcessor()
    word_processor = WordProcessor(image_height=config.IMAGE_HEIGHT,
                                   image_width=config.IMAGE_WIDTH,
                                   position_mode=PositionMode.Left)
    
    words = line_processor(image=image)

    import matplotlib.pyplot as plt
    _, axs = plt.subplots(nrows=len(words), ncols=1)
    for i, word in enumerate(words):
        axs[i].imshow(word, cmap='gray')
    plt.show()
    words_processed = [word_processor(word) for word in words]

    words_processed = np.stack(words_processed, axis=0)
    words_processed = torch.FloatTensor(words_processed).unsqueeze(dim=1).to(device)

    handler.model.eval()
    with torch.no_grad():
        predicts = handler.model.forward(words_processed)
    labels = [convert_to_word(best_path(label)) for label in predicts]
    print(' '.join(labels))
