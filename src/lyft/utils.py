import torch

cmd_codes = dict(m=0, l=1, c=2, a=3, EOS=4, SOS=5, z=6)
COLOR_IDXS = slice(1,6)


def linear_cmd_to_tensor(cmd_index, end_position: tuple, start_position: tuple = None, pad=-1):
    start_pos = start_position if start_position is not None else (0, 0)
    return torch.tensor(
        [cmd_index, *([pad] * 5), start_pos[0], start_pos[1], *([pad] * 4), end_position[0], end_position[1]])


def linear_path_to_tensor(path, pad=-1):
    return torch.stack([linear_cmd_to_tensor(cmd_codes['m'], path[0], pad=pad)] + [
        linear_cmd_to_tensor(cmd_codes['l'], path[i], path[i - 1], pad=pad) for i in range(1, len(path))])


def apply_colors(paths, colors, idxs: slice = COLOR_IDXS):
    colors = colors if colors is not None else [-1] * len(paths)
    for i in range(len(paths)):
        paths[i][:, idxs] = colors[i]
    return paths