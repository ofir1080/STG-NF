import numpy as np
import torch
import argparse
import os

from models.STG_NF.model_pose import STG_NF
from args import init_parser_single, init_sub_args_single
from dataset import get_dataset_and_loader
from utils.train_utils import init_model_params
from utils.scoring_utils_infer import get_video_scores_with_smooth
from tqdm import tqdm


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
    

if __name__ == '__main__':
    main_one_video()