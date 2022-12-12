import random
import numpy as np
import torch
import argparse
import os

from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args, init_parser_single, init_sub_args_single
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset, get_video_scores_with_smooth
from utils.train_utils import calc_num_of_params
from tqdm import tqdm


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        writer = SummaryWriter()
        trainer.train(log_writer=writer)
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test()
    auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
    print("-------------------------------------------------------\n\n")

    print("----------------Unofficall logs--------------------------")
    print(scores.shape, normality_scores.shape)


def test(model, args, loader):
    model.eval()
    pbar = tqdm(loader["test"])
    probs = torch.empty(0).to(args.device)
    print("Starting Test Eval")
    for _, data_arr in enumerate(pbar):
        data = [data.to(args.device, non_blocking=True) for data in data_arr]
        score = data[-2].amin(dim=-1)
        if args.model_confidence:
            samp = data[0]
        else:
            samp = data[0][:, :2]
        with torch.no_grad():
            z, nll = model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
        if args.model_confidence:
            nll = nll * score
        probs = torch.cat((probs, -1 * nll), dim=0)
    prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
    return prob_mat_np


def main_one_video():
    parser = argparse.ArgumentParser(prog="STG-NF-SINGLE")
    parser = init_parser_single()
    args = parser.parse_args()
    args, model_args, frames_num = init_sub_args_single(args)
    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=None, only_test=True)
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    model.to(args.device)
    checkpoint = torch.load(pretrained)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.set_actnorm_init()
    normality_scores = test(model, args, loader)

    scores_arr, dp_scores_np, dp_scores_smoothed, dp_scores_pp_np, score_ids_arr = get_video_scores_with_smooth(normality_scores, dataset["test"].metadata, frames_num, args=args)

    scores_path = f"singal_data/{args.dataset}/output/scores/{args.scores_file_name}" if '/' not in args.scores_file_name else args.scores_file_name
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    np.savez(scores_path, dp_scores_np=dp_scores_np, dp_scores_smoothed=dp_scores_smoothed, dp_scores_pp_np=dp_scores_pp_np, score_ids_arr=score_ids_arr) 
    # print(scores)
    

if __name__ == '__main__':
    # main()
    main_one_video()