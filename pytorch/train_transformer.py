import argparse
import time
from datetime import datetime
import numpy as np
import random
import json
import logging
import torch
import os, sys
from bleu_transformer import evaluate_transformer
from torch.utils import checkpoint

parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)  # add parent folder to path so as to import common modules
from helper import timeSince, sent2indexes, indexes2sent
import models
import configs
import data_loader
from data_loader import APIDataset, APIDataset, load_dict, load_vecs
from metrics import Metrics
from sample import evaluate
from modules import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package
from transformer import Transformer

def train(args):
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    # LOG #
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG,
                        format="%(message)s")  # ,format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    tb_writer = None
    if args.visual:
        # make output directory if it doesn't already exist
        os.makedirs(f'./output/{args.model}/{args.expname}/{timestamp}/models', exist_ok=True)
        os.makedirs(f'./output/{args.model}/{args.expname}/{timestamp}/temp_results', exist_ok=True)
        fh = logging.FileHandler(f"./output/{args.model}/{args.expname}/{timestamp}/logs.txt")
        # create file handler which logs even debug messages
        logger.addHandler(fh)  # add the handlers to the logger
        tb_writer = SummaryWriter(f"./output/{args.model}/{args.expname}/{timestamp}/logs/")
        # save arguments
        json.dump(vars(args), open(f'./output/{args.model}/{args.expname}/{timestamp}/args.json', 'w'))

    # Device #
    if args.gpu_id < 0:
        device = torch.device("cuda")
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id > -1 else "cpu")
    print(device)
    n_gpu = torch.cuda.device_count() if args.gpu_id < 0 else 1
    print(f"num of gpus:{n_gpu}")
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def save_model(model, itr, timestamp):
        """Save model parameters to checkpoint"""
        os.makedirs(f'./output/transformer/{args.expname}/{timestamp}/models', exist_ok=True)
        ckpt_path = f'./output/transformer/{args.expname}/{timestamp}/models/model_itr{itr}.pkl'
        print(f'Saving model parameters to {ckpt_path}')
        torch.save(model.state_dict(), ckpt_path)

    def load_model(model, itr, timestamp):
        """Load parameters from checkpoint"""
        ckpt_path = f'./output/transformer/{args.expname}/202012211035/models/model_itr{itr}.pkl'
        print(f'Loading model parameters from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path))

    def make_mask(src_input, trg_input, device):
        pad_id = 0
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)
        nopeak_mask = torch.ones([1, 50, 50], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false
        return e_mask, d_mask


    config = getattr(configs, 'config_' + args.model)()

    ###############################################################################
    # Load data
    ###############################################################################
    train_set = APIDataset(args.data_path + 'train.desc.h5', args.data_path + 'train.apiseq.h5', config['max_sent_len'])
    valid_set = APIDataset(args.data_path + 'test.desc.h5', args.data_path + 'test.apiseq.h5', config['max_sent_len'])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=1)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=1)
    print("Loaded data!")

    ###############################################################################
    # Define the models
    ###############################################################################
    model = Transformer(10000, 10000).to(device)
    print(model)
    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################
    no_decay = ['bias', 'LayerNorm.weight']
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # do not forget to modify the number when dataset is changed
    criterion = torch.nn.NLLLoss()
    ###############################################################################
    # Training
    ###############################################################################
    logger.info("Training...")
    itr_global = 1
    start_epoch = 1 if args.reload_from == -1 else args.reload_from + 1
    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start_time = time.time()
        itr_start_time = time.time()
        # shuffle (re-define) data between epochs
        for batch in train_loader:  # loop through all batches in training data
            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            old_src_input, src_lens, trg_input, src_lens = batch_gpu
            src_input = torch.zeros_like(old_src_input)
            src_input[:, :-1] = old_src_input[:, 1:50]
            trg_output = torch.zeros_like(trg_input)
            trg_output[:, :-1] = trg_input[:, 1:50]
            trg_input[trg_input == 2] = 0
            # print("src_input:\n", src_input[0])
            # print("trg_input:\n", trg_input[0])
            # print("trg_output:\n", trg_output[0])

            e_mask, d_mask = make_mask(src_input, trg_input, device)

            output = model(src_input, trg_input, e_mask, d_mask)

            trg_output_shape = trg_output.shape
            optim.zero_grad()
            loss = criterion(
                output.view(-1, 10000),
                trg_output.view(trg_output_shape[0] * trg_output_shape[1])
            )
            loss.backward()
            optim.step()

            del src_input, trg_input, trg_output, e_mask, d_mask, output,old_src_input,src_lens
            torch.cuda.empty_cache()
            # after one batch,log
            if itr_global % args.log_every == 0:

                elapsed = time.time() - itr_start_time
                log = 'Transformer-%s|@gpu%d epo:[%d/%d] iter:%d step_time:%ds loss:%f' \
                % (args.expname, args.gpu_id, epoch, config['epochs'], itr_global, elapsed, loss)
                logger.info(log)
                itr_start_time = time.time()

            if itr_global % args.eval_every == 0:
                # evaluate bleu score
                model.eval()
                save_model(model, itr_global, timestamp)
                valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)
                vocab_api = load_dict(args.data_path + 'vocab.apiseq.json')
                vocab_desc = load_dict(args.data_path + 'vocab.desc.json')
                metrics=Metrics()
                os.makedirs(f'./output/transformer/{args.expname}/{timestamp}/temp_results', exist_ok=True)
                f_eval = open(f"./output/transformer/{args.expname}/{timestamp}/temp_results/iter{itr_global}.txt", "w")
                evaluate_transformer(model,metrics, valid_loader, vocab_desc, vocab_api, f_eval)

            itr_global += 1
    # save_model(model, itr_global, timestamp)  # save model after each epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepAPI Pytorch')
    # Path Arguments
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='RNNEncDec', help='model name: RNNEncDec')
    parser.add_argument('--expname', type=str, default='basic',
                        help='experiment name, for disinguishing different parameter settings')
    parser.add_argument('-v', '--visual', action='store_true', default=False,
                        help='visualize training status in tensorboard')
    parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')

    # Evaluation Arguments
    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.add_argument('--log_every', type=int, default=10, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=1000, help='interval to validation')
    parser.add_argument('--eval_every', type=int, default=5000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True  # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True  # fix the random seed in cudnn

    train(args)
