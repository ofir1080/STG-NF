import numpy as np
from utils.scoring_utils import smooth_scores

def get_video_scores_with_smooth(score, metadata, frames_num, args=None):
    dp_scores_np, dp_scores_smoothed, dp_scores_pp_np, score_ids_arr = get_video_scores(score, metadata, frames_num, args=args)
    scores_arr = smooth_scores([dp_scores_smoothed])
    scores_arr = np.concatenate(scores_arr)
    return scores_arr, dp_scores_np, dp_scores_smoothed, dp_scores_pp_np, score_ids_arr


def get_video_scores(scores, metadata, frames_num, args=None):
    metadata_np = np.array(metadata)
    clip_score, clip_ppl_score_arr, fig_score_id = get_clip_score_single(scores, frames_num, metadata_np, metadata, args)

    scores_smoothed_np = score_align(clip_score)
    return clip_score, scores_smoothed_np, clip_ppl_score_arr, fig_score_id



def get_clip_score_single(scores, frames_num, metadata_np, metadata, args):
    clip_metadata = metadata
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    scores_zeros = np.ones(frames_num) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(metadata_np[:, 2] == person_id)[0]
        pid_scores = scores[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amin(clip_ppl_score_arr, axis=0)
    # fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmin(clip_ppl_score_arr, axis=0)]

    return clip_score, clip_ppl_score_arr, list(clip_fig_idxs)

def score_align(scores_np):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1*np.inf] = scores_np[scores_np != -1*np.inf].min()
    return scores_np