import torch


class RasterModel(torch.nn.Module):
    def __init__(self, config: dict = None, future_len=None, in_channels=None, modes: int = 1):
        super().__init__()
        self.modes = modes
        self.in_channels = in_channels
        if self.in_channels is None and config is not None:
            self.in_channels = (config["model_params"]["history_num_frames"] + 1) * 2 + 3
            if config['raster_params']['map_type'] == 'semantic_debug':
                self.in_channels = 3
            if config['raster_params']['map_type'] == 'box_debug':
                self.in_channels = (config["model_params"]["history_num_frames"] + 1) * 2
        self.future_len = future_len
        if self.future_len is None and config is not None:
            self.future_len = config["model_params"]["future_num_frames"] // config["model_params"]["future_step_size"]
        self.num_preds = self.modes * 2 * self.future_len
        self.out_dim = self.num_preds + (self.modes if self.modes != 1 else 0)

    def _forward(self, x):
        return self.model(x)

    def forward(self, x):
        res = self._forward(x)
        if type(x) is list:
            if type(x[0]) is list:
                bs = x[0][0].shape[0]
            else:
                bs = x[0].shape[0]
        else:
            bs = x.shape[0]
        if self.modes != 1:
            pred, conf = torch.split(res, self.num_preds, dim=1)
            pred = pred.view(bs, self.modes, self.future_len, 2)
            conf = torch.softmax(conf, dim=1)
            return pred, conf
        return res.view(bs, 1, self.future_len, 2), res.new_ones((bs, 1))
