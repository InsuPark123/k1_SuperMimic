"""Manually create the same K1 research artifacts as training and playback."""
from pathlib import Path

import tyro

from supermimic.deployment import SingleInputActor, export_k1_policy, motion_csv_rows, prepare_k1_params
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robot import ROBOT


def main(ckpt: Path, npz_dir: Path, out: Path) -> None:
    if ROBOT != 'k1':
        raise ValueError('Set SUPERMIMIC_ROBOT=k1 before exporting')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing directory: {out}')
    actor, cfg = load_policy_from_checkpoint(ckpt, 'cpu')
    prepare_k1_params(out, cfg, discover_npz(npz_dir))
    onnx = export_k1_policy(out, ckpt, actor, cfg)
    print(f'Created research-draft package: {out}; ONNX: {onnx}; NOT ROBOT-DEPLOYABLE')


if __name__ == '__main__':
    tyro.cli(main)
