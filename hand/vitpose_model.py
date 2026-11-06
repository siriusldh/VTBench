from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(
    ROOT_DIR, 'mmpose1x_configs/td-hm_ViTPose-huge_coco-wholebody-256x192.py')
CKPT_PATH = os.path.join(
    ROOT_DIR, '_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth')


def _remap_vitpose_state_dict(sd: dict) -> dict:
    """Map original ViTPose (mmpose-0.x style) keys to mmpose 1.x keys."""
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith('backbone.last_norm.'):
            nk = nk.replace('backbone.last_norm.', 'backbone.ln1.')
        elif nk.startswith('backbone.patch_embed.proj.'):
            nk = nk.replace('backbone.patch_embed.proj.',
                            'backbone.patch_embed.projection.')
        elif nk.startswith('backbone.blocks.'):
            nk = nk.replace('backbone.blocks.', 'backbone.layers.')
            nk = nk.replace('.norm1.', '.ln1.').replace('.norm2.', '.ln2.')
            nk = nk.replace('.mlp.fc1.', '.ffn.layers.0.0.')
            nk = nk.replace('.mlp.fc2.', '.ffn.layers.1.')
        elif nk.startswith('keypoint_head.'):
            nk = nk.replace('keypoint_head.', 'head.')
        if nk == 'backbone.pos_embed' and v.shape[1] == 193:
            # drop cls-token position embedding (model uses no cls token)
            v = v[:, 1:, :]
        new_sd[nk] = v
    return new_sd


class ViTPoseModel(object):
    def __init__(self, device: str | torch.device):
        from mmpose.apis import init_model
        self.device = torch.device(device)
        self.model = init_model(CONFIG_PATH, checkpoint=None,
                                device=str(self.device))
        sd = torch.load(CKPT_PATH, map_location='cpu')
        sd = sd.get('state_dict', sd)
        new_sd = _remap_vitpose_state_dict(sd)
        missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
        real_missing = [k for k in missing if not k.endswith('num_batches_tracked')]
        if real_missing:
            print(f'[ViTPose] missing keys: {real_missing[:10]} ... ({len(real_missing)})')
        if unexpected:
            print(f'[ViTPose] unexpected keys: {unexpected[:10]} ... ({len(unexpected)})')
        self.model.eval()

    @torch.no_grad()
    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        """Return list of dicts with 'keypoints' (133, 3) like mmpose-0.x."""
        from mmpose.apis import inference_topdown
        bboxes = det_results[0][:, :4]
        results = inference_topdown(self.model, image, bboxes=bboxes,
                                    bbox_format='xyxy')
        out = []
        for r in results:
            kpts = r.pred_instances.keypoints[0]          # (133, 2)
            scores = r.pred_instances.keypoint_scores[0]  # (133,)
            out.append({
                'bbox': r.pred_instances.bboxes[0],
                'keypoints': np.concatenate(
                    [kpts, scores[:, None]], axis=-1),
            })
        return out

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float = 0.5,
        kpt_score_threshold: float = 0.3,
        vis_dot_radius: int = 4,
        vis_line_thickness: int = 1,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        return out, image
