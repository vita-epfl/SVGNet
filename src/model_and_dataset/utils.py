import torch
import numpy as np
import csv
from itertools import chain
from typing import Iterator, List, Optional

MAX_MODES = 3



# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
def neg_multi_log_likelihood(
        gt: torch.Tensor, pred: torch.Tensor, confidences: torch.Tensor, avails: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    batch_size, num_modes, future_len, num_coords = pred.shape

    # convert to (batch_size, num_modes, future_len, num_coords)
    if len(gt.shape) != len(pred.shape):
        gt = torch.unsqueeze(gt, 1)  # add modes
    if avails is not None:
        avails = avails[:, None, :, None]  # add modes and cords
        # error (batch_size, num_modes, future_len)
        error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability
    else:
        error = torch.sum((gt - pred) ** 2, dim=-1)  # reduce coords and use availability
    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    return error.reshape(-1)

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/csv_utils.py
def _generate_coords_keys(future_len: int, mode_index: int = 0) -> List[str]:
    """
    Generate keys like coord_x00, coord_y00... that can be used to get or set value in CSV.
    Two keys for each mode and future step.

    Args:
        future_len (int): how many prediction the data has in the future
        mode_index (int): what mode are we reading/writing

    Returns:
        List[str]: a list of keys
    """
    return list(
        chain.from_iterable([[f"coord_x{mode_index}{i}", f"coord_y{mode_index}{i}"] for i in range(future_len)])
    )

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/csv_utils.py
def _generate_confs_keys() -> List[str]:
    """
    Generate modes keys (one per mode)

    Returns:
        List[str]: a list of keys
    """
    return [f"conf_{i}" for i in range(MAX_MODES)]

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/csv_utils.py
def write_pred_csv_header(
        csv_path: str,
        future_len: int,
):

    coords_keys_list = [_generate_coords_keys(future_len, mode_index=idx) for idx in range(MAX_MODES)]
    confs_keys = _generate_confs_keys()

    fieldnames = ["idx", "grads/semantics", "grads/vehicles", "grads/total", "nll", "loss","timestamp", "track_id"] + confs_keys # all confidences before coordinates
    for coords_labels in coords_keys_list:
        fieldnames.extend(coords_labels)
    writer = csv.DictWriter(open(csv_path+"/full_result.csv", "w"), fieldnames)
    writer.writeheader()

    return writer, confs_keys, coords_keys_list

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/csv_utils.py
def write_pred_csv_data(
        writer: csv.DictWriter,
        confs_keys: list,
        coords_keys_list: list,
        timestamps: np.ndarray,
        track_ids: np.ndarray,
        result,
) -> None:

    coords = result["pred"]
    confs = result["conf"].cpu().detach().numpy().copy()
    assert len(coords.shape) in [3, 4]

    if len(coords.shape) == 3:
        assert confs is None  # no conf for the single-mode case
        coords = np.expand_dims(coords, 1)  # add a new axis for the multi-mode
        confs = np.ones((len(coords), 1))  # full confidence

    num_example, num_modes, future_len, num_coords = coords.shape
    assert num_coords == 2
    assert timestamps.shape == track_ids.shape == (num_example,)
    assert confs is not None and confs.shape == (num_example, num_modes)
    assert np.allclose(np.sum(confs, axis=-1), 1.0)
    assert num_modes <= MAX_MODES

    # generate always a fixed size json for MAX_MODES by padding the arrays with zeros
    coords_padded = np.zeros((num_example, MAX_MODES, future_len, num_coords), dtype=coords.dtype)
    coords_padded[:, :num_modes] = coords
    confs_padded = np.zeros((num_example, MAX_MODES), dtype=confs.dtype)
    confs_padded[:, :num_modes] = confs

    for idx, gs, gv, gt, nll, loss, timestamp, track_id, coord, conf in zip(result["idx"].cpu().numpy().copy(), result["grads/semantics"].cpu().numpy().copy(),
                                                                            result["grads/vehicles"].cpu().numpy().copy(), result["grads/total"].cpu().numpy().copy(),
                                                                            result["nll"].cpu().detach().numpy().copy(), result["loss"].cpu().detach().numpy().copy(),
                                                                            timestamps.cpu().numpy().copy(), track_ids.cpu().numpy().copy(), coords_padded, confs_padded):
        line = {"idx": idx, "grads/semantics": gs, "grads/vehicles": gv, "grads/total": gt, "nll": nll, "loss": loss, "timestamp": timestamp, "track_id": track_id}
        line.update({key: con for key, con in zip(confs_keys, conf)})

        for idx in range(MAX_MODES):
            line.update({key: f"{cor:.5f}" for key, cor in zip(coords_keys_list[idx], coord[idx].reshape(-1))})

        writer.writerow(line)
