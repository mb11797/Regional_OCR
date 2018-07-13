import torch
import os
import argparse
import numpy as np
from model

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # data loader
    images = np.load('img_data.npy')
    labels = np.load('labels.npy')

    # Build the models







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, deafult='models/', help='path for saving trained models')
    parser.add_argument('--num_classes')
    args = parser.parse_args()
    print(args)
    main(args)











