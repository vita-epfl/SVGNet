import types
import typing as th
from l5kit.data.filter import filter_tl_faces_by_status, filter_agents_by_track_id, filter_agents_by_labels
from l5kit.data.map_api import MapAPI
from l5kit.data import get_frames_slice_from_scenes
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points, yaw_as_rotation33
from l5kit.rasterization.semantic_rasterizer import elements_within_bounds, cv2_subpixel
from l5kit.rasterization.box_rasterizer import get_ego_as_agent
from typing import List, Optional
import numpy as np
import warnings
import torch
import cv2
from collections import defaultdict
from src.lyft.utils import linear_path_to_tensor

CV2_SHIFT = 8
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT
AGENT_TYPE = 4
EGO_TYPE = 5


def normalize_line(line):
    return line / CV2_SHIFT_VALUE


def path_type_to_number(path_type):
    if path_type == 'black':
        return 0
    if path_type == 'green':
        return 1
    if path_type == 'yellow':
        return 2
    if path_type == 'red':
        return 3
    if path_type == 'agent':
        return AGENT_TYPE
    if path_type == 'ego':
        return EGO_TYPE


def lane_color(path_number):
    if path_number == 0:
        return 'black'
    if path_number == 1:
        return 'green'
    if path_number == 2:
        return 'yellow'
    if path_number == 3:
        return 'red'
    if path_number == AGENT_TYPE:
        return 'blue'
    if path_number == EGO_TYPE:
        return 'cyan'


def crop_tensor(vector, raster_size):
    vector = vector[(vector[:, 0] >= 0.) * (vector[:, 0] <= raster_size[0])]
    vector = vector[(vector[:, 1] >= 0.) * (vector[:, 1] <= raster_size[1])]
    vector[:, 0] = (vector[:, 0] / raster_size[0]) * 24
    vector[:, 1] = (vector[:, 1] / raster_size[1]) * 24
    return vector


def render_semantic_map(
        self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray = None,
        tl_face_color=True
) -> th.Union[torch.Tensor, dict]:
    """Renders the semantic map at given x,y coordinates.
    Args:
        center_in_world (np.ndarray): XY of the image center in world ref system
        raster_from_world (np.ndarray):
    Returns:
        th.Union[torch.Tensor, dict]
    """
    # filter using half a radius from the center
    raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

    # get active traffic light faces
    if tl_face_color:
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

    # setup canvas
    raster_size = self.render_context.raster_size_px
    res = dict(path=list(), path_type=list())
    for idx in elements_within_bounds(center_in_world, self.bounds_info["lanes"]["bounds"], raster_radius):
        lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

        # get image coords
        lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
        xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], raster_from_world))
        xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], raster_from_world))
        xy_left = normalize_line(xy_left)
        xy_right = normalize_line(xy_right)

        lane_type = "black"  # no traffic light face is controlling this lane
        if tl_face_color:
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"

        for vector in [xy_left, xy_right]:
            vector = crop_tensor(vector, raster_size)
            if len(vector):
                res['path'].append(vector)
                res['path_type'].append(path_type_to_number(lane_type))
    return res


def rasterize_semantic(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
        svg=False, svg_args=None,
):
    if agent is None:
        ego_translation_m = history_frames[0]["ego_translation"]
        ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
    else:
        ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
        ego_yaw_rad = agent["yaw"]

    raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
    world_from_raster = np.linalg.inv(raster_from_world)

    # get XY of center pixel in world coordinates
    center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
    center_in_world_m = transform_point(center_in_raster_px, world_from_raster)
    res = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])

    svg_args = svg_args or dict()
    if svg:
        res['path'] = torch.cat(
            [linear_path_to_tensor(path, svg_args.get('pad_val', -1)) for path in res['path']], 0)
    return res


def add_agents(res_dict, agents):
    for idx, agent in enumerate(agents):
        res_dict[idx].append(agent["centroid"][:2])


def calc_max_grad(path):
    return np.sqrt(np.square(np.diff(path, axis=0)).sum(1)).max()


def is_noisy(path, ref_grad, tolerance=20):
    return (len(path) < 2) or calc_max_grad(path) > (ref_grad + tolerance)


def rasterize_box(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
        svg=False, svg_args=None,
) -> th.Union[dict]:
    # all frames are drawn relative to this one"
    frame = history_frames[0]
    if agent is None:
        ego_translation_m = history_frames[0]["ego_translation"]
        ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
    else:
        ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
        ego_yaw_rad = agent["yaw"]
    svg_args = svg_args or dict()
    raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
    raster_size = self.render_context.raster_size_px
    # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
    res = dict(ego=list(), agents=defaultdict(list))
    for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
        # print('history index', i)
        agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
        # note the cast is for legacy support of dataset before April 2020
        av_agent = get_ego_as_agent(frame).astype(agents.dtype)

        if agent is None:
            add_agents(res['agents'], av_agent)
            res['ego'].append(av_agent[0]["centroid"][:2])
        else:
            agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
            if len(agent_ego) == 0:  # agent not in this history frame
                add_agents(res['agents'], np.append(agents, av_agent))
            else:  # add av to agents and remove the agent from agents
                agents = agents[agents != agent_ego[0]]
                add_agents(res['agents'], np.append(agents, av_agent))
                res['ego'].append(agent_ego[0]["centroid"][:2])
    tolerance = svg_args.get('tolerance', 20.)
    _ego = normalize_line(
        cv2_subpixel(transform_points(np.array(res['ego']).reshape((-1, 2)), raster_from_world)))
    res['ego'] = crop_tensor(_ego, raster_size)
    ego_grad = calc_max_grad(res['ego'])
    res['agents'] = [normalize_line(cv2_subpixel(transform_points(np.array(path).reshape((-1, 2)), raster_from_world))
                                    ) for idx, path in res['agents'].items()]
    res['agents'] = [
        crop_tensor(path, raster_size) for path in res['agents'] if not is_noisy(path, ego_grad, tolerance)]
    res['agents'] = [path for path in res['agents'] if len(path)]

    if svg:
        res['path'] = torch.cat([linear_path_to_tensor(path, svg_args.get('pad_val', -1)) for path in res['agents']
                                 ] + [linear_path_to_tensor(res['ego'], svg_args.get('pad_val', -1))], 0)
        res['path_type'] = [path_type_to_number('agent')] * len(res['agents']) + [path_type_to_number('ego')]
    return res


def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None,
              vehicles=False) -> dict:
    """
    A utility function to get the rasterisation and trajectory target for a given agent in a given frame
    Args:
        scene_index (int): the index of the scene in the zarr
        state_index (int): a relative frame index in the scene
        track_id (Optional[int]): the agent to rasterize or None for the AV
    Returns:
        dict: the rasterised image in (Cx0x1) if the rast is not None, the target trajectory
        (position and yaw) along with their availability, the 2D matrix to center that agent,
        the agent track (-1 if ego) and the timestamp
    """
    frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
    tl_faces = self.dataset.tl_faces
    try:
        if self.cfg["raster_params"]["disable_traffic_light_faces"]:
            tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
    except KeyError:
        warnings.warn(
            "disable_traffic_light_faces not found in config, this will raise an error in the future",
            RuntimeWarning,
            stacklevel=2,
        )
    data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

    target_positions = np.array(data["target_positions"], dtype=np.float32)
    target_yaws = np.array(data["target_yaws"], dtype=np.float32)

    history_positions = np.array(data["history_positions"], dtype=np.float32)
    history_yaws = np.array(data["history_yaws"], dtype=np.float32)

    timestamp = frames[state_index]["timestamp"]
    track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

    result = {
        "target_positions": target_positions,
        "target_yaws": target_yaws,
        "target_availabilities": data["target_availabilities"],
        "history_positions": history_positions,
        "history_yaws": history_yaws,
        "history_availabilities": data["history_availabilities"],
        "world_to_image": data["raster_from_world"],  # TODO deprecate
        "raster_from_world": data["raster_from_world"],
        "raster_from_agent": data["raster_from_agent"],
        "agent_from_world": data["agent_from_world"],
        "world_from_agent": data["world_from_agent"],
        "track_id": track_id,
        "timestamp": timestamp,
        "centroid": data["centroid"],
        "yaw": data["yaw"],
        "extent": data["extent"],
    }

    # when rast is None, image could be None
    if isinstance(data["image"], dict):
        for i, j in data['image'].items():
            result[i] = j

    elif data["image"] is not None:
        # 0,1,C -> C,0,1
        result["image"] = data["image"]
    return result


def rasterize_sem_box(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
        svg=False, svg_args=None,
) -> np.ndarray:
    res_box = self.box_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
    res_sat = self.sat_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
    if not svg:
        return {**res_box, **res_sat}
    svg_args = svg_args or dict()
    res = dict()
    res['path'] = torch.cat(
        [linear_path_to_tensor(path, svg_args.get('pad_val', -1)) for path in res_sat['path']
         ] + [linear_path_to_tensor(path, svg_args.get('pad_val', -1)) for path in res_box['agents']
              ] + [linear_path_to_tensor(res_box['ego'], svg_args.get('pad_val', -1))], 0)
    res['path_type'] = res_sat['path_type'] + [path_type_to_number('agent')] * len(res_box['agents']) + [
        path_type_to_number('ego')]
    return res
