import copy
from pathlib import Path


_base_namespace = {}
_base_path = Path(__file__).resolve().with_name("base_track_map.py")
with _base_path.open("r", encoding="utf-8") as _file:
    exec(_file.read(), _base_namespace, _base_namespace)

plugin = _base_namespace["plugin"]
plugin_dir = _base_namespace["plugin_dir"]
model = copy.deepcopy(_base_namespace["model"])

if "test_cfg" in _base_namespace:
    test_cfg = copy.deepcopy(_base_namespace["test_cfg"])

pts_transformer = model["pts_bbox_head"]["transformer"]
pts_transformer["num_cams"] = 4
pts_transformer["use_cams_embeds"] = False
pts_transformer["encoder"]["transformerlayers"]["attn_cfgs"][1]["num_cams"] = 4
