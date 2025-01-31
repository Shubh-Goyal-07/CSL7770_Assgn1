import argparse
import logging
import os
from utils.train_nn import main as train_nn_main
from utils.train_ml import main as train_ml_main
from utils.train_cnn import main as train_cnn_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, help='Model to train')
    parser.add_argument('--window', type=str, help='Window type to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    args = parser.parse_args()

    if args.window not in ['hann', 'hamming', 'rectangular']:
        logging.error('Invalid window type')
        exit(1)

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename=f'logs/train_{args.model}_{args.window}.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    if args.model == 'nn':
        train_nn_main(args.window, args.epochs)
    elif args.model == 'cnn':
        train_cnn_main(args.window, args.epochs)
    elif args.model in ['svm', 'knn', 'rf', 'dt']:
        train_ml_main(args.window, args.model)
    else:
        logging.error('Invalid model')
        exit(1)