import torch
from l5kit.dataset import AgentDataset as _AgentDataset
from l5kit.rasterization import build_rasterizer as _build_rasterizer

from l5kit.rasterization import SemanticRasterizer, SemBoxRasterizer, BoxRasterizer
from src.lyft.rasterizer import (
    render_semantic_map, rasterize_semantic, rasterize_sem_box, get_frame, rasterize_box)

import functools
import copy
import types
from deepsvg.svglib.svg import SVG, Bbox

def build_rasterizer(config, data_manager):
    map_type = config['raster_params']['map_type']
    config_prime = copy.deepcopy(config)
    base_map_type = config_prime['raster_params']['map_type'] = map_type.replace('svg_', '').replace('tensor_', '')

    rasterizer = _build_rasterizer(config_prime, data_manager)
    tl_face_color = not config['raster_params']['disable_traffic_light_faces']
    svg = map_type.startswith('svg_')
    tensor = map_type.startswith('tensor_')
    if svg or tensor:
        svg_args = config['raster_params'].get('svg_args', dict())
        render_semantics = functools.partial(render_semantic_map, tl_face_color=tl_face_color)
        if isinstance(rasterizer, SemanticRasterizer):
            rasterize_sem = functools.partial(
                rasterize_semantic, svg=svg, svg_args=svg_args)
            rasterizer.render_semantic_map = types.MethodType(render_semantics, rasterizer)
            rasterizer.rasterize = types.MethodType(rasterize_sem, rasterizer)

        if isinstance(rasterizer, SemBoxRasterizer):
            rasterize_sem = functools.partial(rasterize_semantic, svg=False, svg_args=None)
            rasterize_sembox = functools.partial(rasterize_sem_box, svg=svg, svg_args=svg_args)
            rasterize_b = functools.partial(rasterize_box, svg=False, svg_args=svg_args)
            rasterizer.sat_rast.render_semantic_map = types.MethodType(render_semantics, rasterizer.sat_rast)
            rasterizer.sat_rast.rasterize = types.MethodType(rasterize_sem, rasterizer.sat_rast)
            rasterizer.rasterize = types.MethodType(rasterize_sembox, rasterizer)
            rasterizer.box_rast.rasterize = types.MethodType(rasterize_b, rasterizer.box_rast)

        if isinstance(rasterizer, BoxRasterizer):
            rasterize_b = functools.partial(rasterize_box, svg=svg, svg_args=svg_args)
            rasterizer.rasterize = types.MethodType(rasterize_b, rasterizer)

    return rasterizer


def agent_dataset(cfg: dict, zarr_dataset, rasterizer, perturbation=None, agents_mask=None,
                  min_frame_history=10, min_frame_future=1):
    data = _AgentDataset(cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)
    map_type = cfg['raster_params']['map_type']
    svg = map_type.startswith('svg_')
    tensor = map_type.startswith('tensor_')
    if svg or tensor:
        data.get_frame = types.MethodType(get_frame, data)
    return data