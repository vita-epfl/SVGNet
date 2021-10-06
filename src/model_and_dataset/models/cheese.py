import torch
from abc import ABC


class MaskedLinear(torch.nn.Linear, ABC):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask_weights=True, mask_bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask_weights = mask_weights
        self.mask_bias = mask_bias

    @staticmethod
    def generate_mask(mask_lens, length):
        idx = torch.arange(length).to(mask_lens.device)
        return (idx.reshape(-1, length) < mask_lens.reshape(-1, 1)).float().to(mask_lens.device)

    @staticmethod
    def generate_bias_mask(mask_lens):
        return (mask_lens != 0).view(-1, 1)

    def forward(self, inputs, mask_lens=None, mask=None, mask_bias=None):
        # linear
        if self.mask_weights:
            mask = mask if mask is not None else self.generate_mask(mask_lens, self.in_features)
            linear = (inputs.view(-1, 1, inputs.shape[1]) @ (self.weight * mask.view(inputs.shape[0], 1, -1)).permute(
                0, 2, 1)).squeeze(1)
        else:
            linear = inputs @ self.weight.permute(1, 0)
        # bias
        if self.mask_bias:
            if mask_bias is None:
                mask_bias = self.generate_bias_mask(mask_lens)
            bias = 0 if self.bias is None else self.bias.view(1, -1).repeat(inputs.shape[0], 1) * mask_bias
        else:
            bias = 0 if self.bias is None else self.bias
        return linear + bias


class MLP(torch.nn.Module, ABC):
    def __init__(self, in_features, num_features, out_features, num_layers, bias=True, residual=False, act_last=False,
                 masked=False):
        super().__init__()
        self.in_features = in_features
        self.num_features = num_features
        self.out_features = out_features
        self.act_last = act_last
        self.num_layers = num_layers
        self.residual = residual
        self.masked = masked
        self.act = torch.nn.ReLU(inplace=True)
        for i in range(num_layers):
            in_dim = num_features if i else in_features
            out_dim = out_features if i == num_layers - 1 else num_features
            mask_weights = (not i) and masked
            mask_bias = masked
            layer = MaskedLinear(in_dim, out_dim, bias=bias, mask_weights=mask_weights, mask_bias=mask_bias)
            if self.act_last or i != num_layers - 1:
                torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity='relu')
            setattr(self, f'layer{i}', layer)

    def forward(self, inputs, mask_lens=None, mask=None, mask_bias=None):
        if self.masked:
            mask = mask if mask is not None else MaskedLinear.generate_mask(mask_lens, self.in_features)
            mask_bias = mask_bias if mask_bias is not None else MaskedLinear.generate_bias_mask(mask_lens)
        in_features = self.in_features
        for i in range(self.num_layers):
            layer = getattr(self, f'layer{i}')
            out_features = layer.out_features
            result = layer(inputs, mask_lens, mask, mask_bias)
            if i != self.num_layers - 1 or self.act_last:
                result = self.act(result)
            inputs = result + (inputs if self.residual and out_features == in_features else 0)
            in_features = out_features
        return inputs


class Albert(torch.nn.Module, ABC):
    def __init__(self, layer, num_layers, layer_norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.layer = layer
        self.layer_norm = layer_norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for i in range(self.num_layers):
            src = self.layer(src, src_mask, src_key_padding_mask)
        if self.layer_norm is not None:
            src = self.layer_norm(src)
        return src


class Cheese(torch.nn.Module, ABC):
    def __init__(self, history_num=20, scene_num=170, agents_num=20, modes=1, future_len=30,
                 num_layers=20, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu',
                 layer_norm=False, albert=True, label_embedding=False, hist_residual_transform=False,
                 traj_ff_dim=40, traj_num_layers=5, scene_ff_dim=340, scene_num_layers=5,
                 mask_hist=False, mask_agents=False, mask_scene=True,
                 separated_transforms=False, cat_transforms=True,
                 resnet_num_layers=4, hist_final_cat=True, hist_final_cat_dim=64, hist_final_cat_layers=2,
                 final_dim=128, final_num_layers=3,
                 ):
        super().__init__()
        self.history, self.scene, self.agents = history_num is not None, scene_num is not None, agents_num is not None
        self.hist_final_cat = hist_final_cat and self.history
        self.hist_residual_transform = hist_residual_transform and self.history
        self.cat_transforms = cat_transforms
        self.label_embedding = label_embedding
        self.separated_transforms = separated_transforms

        self.modes = modes
        self.future_len = future_len
        self.num_preds = self.modes * 2 * self.future_len
        self.out_dim = self.num_preds + (self.modes if self.modes != 1 else 0)

        # embedding
        if self.history:
            self.embed_hist = MLP(history_num * 2, traj_ff_dim, d_model, traj_num_layers, masked=mask_hist,
                                  residual=True, act_last=False)
        if self.hist_final_cat:
            self.hist_cat_final_embedding = MLP(
                history_num * 2, hist_final_cat_dim, hist_final_cat_dim, hist_final_cat_layers, masked=mask_hist,
                residual=False, act_last=False)
        if self.agents:
            self.embed_agents = MLP(history_num * 2, traj_ff_dim, d_model, traj_num_layers, masked=mask_agents,
                                    residual=True, act_last=False)
        if self.scene:
            self.embed_scene = MLP(scene_num * 2, scene_ff_dim, d_model, scene_num_layers,
                                   masked=mask_scene, residual=True, act_last=False)
        if self.label_embedding:
            self.label = torch.nn.Embedding(self.history + self.scene + self.agents, d_model)
        # transformer
        layer_norm = torch.nn.LayerNorm(d_model) if layer_norm else None
        transform_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
            nhead=nhead, activation=activation
        )

        self.transform = torch.nn.TransformerEncoder(
            transform_layer, num_layers, layer_norm) if not albert else Albert(transform_layer, num_layers, layer_norm)

        if self.separated_transforms:
            self.mix_results = torch.nn.Linear(
                ((self.history + self.scene + self.agents) * d_model if cat_transforms else d_model), d_model,
                bias=True)
        self.resnet = MLP(d_model, d_model, final_dim, resnet_num_layers + 1, residual=True, act_last=False)
        self.final = MLP(final_dim + hist_final_cat_dim if self.hist_final_cat else final_dim, final_dim, self.out_dim,
                         final_num_layers, residual=False, act_last=False)

    def _forward(self, history_positions=None, scene=None, scene_lens=None, agents=None, agents_lens=None):
        # embedding
        viz_mask, poly_lines = [], []
        item_id, idx = 0, 0
        if self.history:
            hist_idx = idx
            idx += 1
            hist = history_positions.reshape(history_positions.shape[0], -1)
            label = self.label(torch.LongTensor([item_id]).to(hist.device)) if self.label_embedding else 0
            hist_embed = self.embed_hist(hist.reshape(-1, hist.shape[-1])).reshape(hist.shape[0], 1, -1).permute(
                1, 0, 2) + label
            if self.hist_final_cat:
                hist_cat = self.hist_cat_final_embedding(hist).reshape(hist.shape[0], -1)
            poly_lines.append(hist_embed)
            viz_mask.append(torch.zeros(hist.shape[0], 1).to(hist.device))
            item_id += 1

        if self.scene:
            scene = scene  # batch['padded_cntr_lines']
            scene_lens = scene_lens
            scene_idx = slice(idx, idx + scene.shape[1])
            idx += scene.shape[1]
            scene_bias_mask = MaskedLinear.generate_bias_mask(scene_lens) if self.embed_scene.masked else None
            label = self.label(torch.LongTensor([item_id]).to(scene.device)) if self.label_embedding else 0
            scene_embed = self.embed_scene(
                scene.reshape(-1, scene.shape[-1]), scene_lens, mask_bias=scene_bias_mask
            ).reshape(scene.shape[0], scene.shape[1], -1).permute(1, 0, 2) + label
            poly_lines.append(scene_embed)
            viz_mask.append((1 - scene_bias_mask.float().reshape(-1, scene.shape[1])).to(scene.device))

        if self.agents:
            agents = agents  # batch['padded_cntr_lines']
            agents_lens = agents_lens
            agents_idx = slice(idx, idx + agents.shape[1])
            idx += agents.shape[1]
            agents_bias_mask = MaskedLinear.generate_bias_mask(agents_lens) if self.embed_agents.masked else None
            label = self.label(torch.LongTensor([item_id]).to(agents.device)) if self.label_embedding else 0
            agents_embed = self.embed_agents(
                agents.reshape(-1, agents.shape[-1]), agents_lens, mask_bias=agents_bias_mask
            ).reshape(agents.shape[0], agents.shape[1], -1).permute(1, 0, 2) + label
            poly_lines.append(agents_embed)
            viz_mask.append(
                (1 - agents_bias_mask.float().reshape(-1, agents.shape[1])).to(agents.device) \
                    if agents_bias_mask is not None else \
                    torch.zeros(agents.shape[0], agents.shape[1]).to(hist.device)
            )
        viz_mask = torch.cat(viz_mask, dim=1)
        poly_lines = torch.cat(poly_lines)
        transformed = self.transform(poly_lines, src_key_padding_mask=viz_mask.type(torch.bool)) * (
                1 - viz_mask).permute(1, 0).unsqueeze(-1)

        if self.separated_transforms:
            results = []
            if self.history:
                transformed_hist = (transformed[hist_idx, ...] + (hist_embed[0] if self.hist_residual_transform else 0)
                                    ).reshape(hist.shape[0], -1) / (1 + self.hist_residual_transform)
                results.append(transformed_hist)
            if self.scene:
                transformed_scene = transformed[scene_idx, ...].sum(0) / (
                        1 - viz_mask.permute(1, 0)[scene_idx, ...]).sum(
                    0).reshape(-1, 1)
                results.append(transformed_scene)
            if self.agents:
                transformed_agents = transformed[agents_idx, ...].sum(0) / (
                        1 - viz_mask.permute(1, 0)[agents_idx, ...]).sum(0).reshape(-1, 1)
                results.append(transformed_agents)

            results = torch.cat(results, dim=1) if self.cat_transforms else sum(results) / (
                    self.history + self.scene + self.agents)
            results = self.mix_results(results)
        else:
            results = transformed.sum(0) / (1 - viz_mask.permute(1, 0)).sum(0).reshape(-1, 1)
        results = self.resnet(results)
        if self.hist_final_cat:
            results = torch.cat([results, hist_cat], dim=1)
        return self.final(results)

    def forward(self, x):
        history_positions, scene, scene_lens, agents, agents_lens = x
        res = self._forward(history_positions, scene, scene_lens, agents, agents_lens)
        bs = history_positions.shape[0]
        if self.modes != 1:
            pred, conf = torch.split(res, self.num_preds, dim=1)
            pred = pred.reshape(bs, self.modes, self.future_len, 2)
            conf = torch.softmax(conf, dim=1)
            return pred, conf
        return res.reshape(bs, 1, self.future_len, 2), res.new_ones((bs, 1))
