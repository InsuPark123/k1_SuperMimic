"""K1 run artifacts. The stock robot runtime cannot consume these research drafts."""
from __future__ import annotations

import hashlib
import copy
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from supermimic.data.motion_convert import K1_JOINT_NAMES
from supermimic.envs.k1_config import K1_REV1_ARTICULATION_CFG, K1_REV1_INIT_STATE


class SingleInputActor(torch.nn.Module):
    def __init__(self, policy, history_dim):
        super().__init__()
        self.policy = policy
        self.history_dim = history_dim

    def forward(self, observations):
        return self.policy(observations[:, :self.history_dim], observations[:, self.history_dim:])


def motion_csv_rows(body_pos, body_quat, joint_pos, root_idx):
    return np.concatenate((body_pos[:, root_idx, :], body_quat[:, root_idx, :][:, [1, 2, 3, 0]], joint_pos), axis=1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def joint_parameters():
    init = K1_REV1_INIT_STATE.joint_pos
    if not isinstance(init, dict):
        raise ValueError('Unexpected K1 default-joint configuration')
    properties, scales, offsets = {}, [], []
    for name in K1_JOINT_NAMES:
        actuators = [a for a in K1_REV1_ARTICULATION_CFG.actuators
                     if any(re.fullmatch(expr, name) for expr in a.target_names_expr)]
        defaults = [value for expr, value in init.items() if re.fullmatch(expr, name)]
        if len(actuators) != 1 or len(defaults) > 1:
            raise ValueError(f'Ambiguous K1 actuator or default for {name}')
        a = actuators[0]
        offset = float(defaults[0] if defaults else 0.0)
        properties[name] = dict(default_position=offset, stiffness=float(a.stiffness),
                                damping=float(a.damping), position_limit=None)
        scales.append(float(0.25 * a.effort_limit / a.stiffness))
        offsets.append(offset)
    return properties, scales, offsets


def _config(cfg):
    return asdict(cfg) if is_dataclass(cfg) else cfg


def prepare_k1_params(run_dir, cfg, motion_files):
    """Snapshot training motions and draft runtime parameters without overwriting."""
    cfg = _config(cfg)
    files = [Path(f) for f in motion_files]
    if not files or any(p.suffix != '.npz' for p in files):
        raise ValueError('K1 artifact generation requires individual motion NPZ files')
    if len({p.stem for p in files}) != len(files):
        raise ValueError('Duplicate motion names')
    history = int(cfg['policy']['history']) * 75
    goal = int(cfg['policy']['goal_dim'])
    if cfg['env'].get('anchor_obs', False) and cfg['env'].get('anchor_ori_obs', False):
        raise ValueError('anchor_obs and anchor_ori_obs are mutually exclusive')
    expected = int(cfg['env']['future_frames']) * 56 + (9 if cfg['env'].get('anchor_obs', False) else 6 if cfg['env'].get('anchor_ori_obs', False) else 0)
    if history != 750 or goal != expected:
        raise ValueError(f'Unexpected K1 policy input dimensions: {history}, {goal}, expected {expected}')
    sources = {p.name: sha256(p) for p in files}
    run_dir = Path(run_dir)
    target = run_dir / 'params'
    if target.exists():
        manifest = target / 'manifest.json'
        if not manifest.is_file():
            raise ValueError(f'existing params directory is unmanaged: {target}')
        existing = json.loads(manifest.read_text())
        if existing.get('motion_sha256') != sources or existing.get('input_sizes') != [history, goal]:
            raise ValueError(f'existing params differ from this run: {target}')
        for name, digest in existing['artifact_sha256'].items():
            if not (target / name).is_file() or sha256(target / name) != digest:
                raise ValueError(f'existing params artifact was modified: {target / name}')
        return target
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.params-', dir=run_dir) as tmp:
        stage = Path(tmp) / 'params'
        stage.mkdir()
        properties, scale, offset = joint_parameters()
        settings = {
            'policy_joints': list(K1_JOINT_NAMES), 'step_dt': 0.02,
            'joint_properties': properties, 'commands': {},
            'actions': {'joint_pos': {'clip': None, 'scale': scale, 'offset': offset}},
            # Fail closed until the deployment runtime implements the exact input terms.
            'observations': {
                'supermimic_policy_history': {'params': {}, 'clip': None, 'scale': [1.0] * history, 'history_length': 1},
                'supermimic_motion_goal': {'params': {}, 'clip': None, 'scale': [1.0] * goal, 'history_length': 1},
            },
        }
        (stage / 'sim2real.yaml').write_text('# DRAFT: custom SuperMimic observation adapter required. DO NOT DEPLOY.\n' + yaml.safe_dump(settings, sort_keys=False))
        (stage / 'README.md').write_text(
            '# Research draft — not robot-deployable\n\n'
            'The stock AI Sapiens runtime does not provide the two SuperMimic observation terms. '
            'The YAML uses unregistered names so that it cannot silently run with incorrect inputs. '
            'Implement and validate the runtime adapter and a real reference-relative torso pose estimator first.\n\n'
            f'The actor input is a single float32 vector: {history} history values followed by {goal} goal values. '
            'The actor already contains its normalization. Output is 23 normalized joint actions; '
            'apply the configured scale and offset once. Control rate is 50 Hz.\n\n'
            'CSV files contain root position, xyzw root quaternion and 23 named joint positions. '
            'The NPZ copies additionally preserve the precomputed reference velocities needed for '
            'the exact motion goal; CSV alone is insufficient for parity.\n'
        )
        for path in files:
            with np.load(path) as d:
                names = tuple(map(str, d['joint_names']))
                if names != K1_JOINT_NAMES or abs(float(d['fps'].reshape(-1)[0]) - 50.0) > 1e-6:
                    raise ValueError(f'{path}: expected K1 joint order and 50 Hz')
                root = list(map(str, d['body_names'])).index('pelvis')
                rows = motion_csv_rows(d['body_pos_w'], d['body_quat_w'], d['joint_pos'], root)
                if not np.isfinite(rows).all():
                    raise ValueError(f'{path}: non-finite motion values')
                header = ','.join(['root_pos_x', 'root_pos_y', 'root_pos_z', 'root_quat_x', 'root_quat_y', 'root_quat_z', 'root_quat_w', *names])
                np.savetxt(stage / (path.stem + '.csv'), rows, delimiter=',', header=header, comments='', fmt='%.9g')
            shutil.copyfile(path, stage / path.name)
        hashes = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
        manifest = {'status': 'NOT_DEPLOYABLE_runtime_adapter_and_pose_estimator_required',
                    'input_sizes': [history, goal], 'motion_fps': 50, 'motion_sha256': sources,
                    'artifact_sha256': hashes}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.rename(stage, target)
    return target


def export_k1_policy(run_dir, ckpt, policy, cfg):
    """Save a versioned ONNX actor and update the managed policy.onnx copy."""
    cfg = _config(cfg)
    history = int(cfg['policy']['history']) * 75
    goal = int(cfg['policy']['goal_dim'])
    if cfg['env'].get('anchor_obs', False) and cfg['env'].get('anchor_ori_obs', False):
        raise ValueError('anchor_obs and anchor_ori_obs are mutually exclusive')
    if history != 750 or goal != int(cfg['env']['future_frames']) * 56 + (9 if cfg['env'].get('anchor_obs', False) else 6 if cfg['env'].get('anchor_ori_obs', False) else 0):
        raise ValueError('Unexpected policy input dimensions')
    ckpt = Path(ckpt)
    ckpt_sha = sha256(ckpt)
    exported = Path(run_dir) / 'exported'
    version = exported / 'checkpoints' / f'{ckpt.stem}-{ckpt_sha[:12]}'
    final = version / 'policy.onnx'
    if version.exists():
        manifest_path = version / 'manifest.json'
        if not final.is_file() or not manifest_path.is_file():
            raise ValueError(f'Incomplete or unmanaged export: {version}')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('checkpoint_sha256') != ckpt_sha or manifest.get('onnx_sha256') != sha256(final):
            raise ValueError(f'Existing export does not match checkpoint: {version}')
    else:
        version.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.onnx-', dir=version.parent) as tmp:
            stage = Path(tmp) / version.name
            stage.mkdir()
            wrapped = SingleInputActor(copy.deepcopy(policy).cpu().eval(), history).eval()
            sample = torch.zeros(1, history + goal, dtype=torch.float32)
            torch.onnx.export(wrapped, (sample,), str(stage / 'policy.onnx'), input_names=['observations'],
                              output_names=['actions'], opset_version=17, dynamo=False)
            import onnx
            onnx.checker.check_model(onnx.load(str(stage / 'policy.onnx')))
            manifest = {'status': 'NOT_DEPLOYABLE_runtime_adapter_and_pose_estimator_required',
                        'checkpoint': str(ckpt.resolve()), 'checkpoint_sha256': ckpt_sha,
                        'onnx_sha256': sha256(stage / 'policy.onnx'), 'input_sizes': [history, goal],
                        'output_size': len(K1_JOINT_NAMES)}
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            os.rename(stage, version)
    exported.mkdir(parents=True, exist_ok=True)
    link = exported / 'policy.onnx'
    latest_manifest = exported / 'manifest.json'
    if link.is_symlink():
        old = os.readlink(link)
        resolved = link.resolve()
        if not old.startswith('checkpoints/') or not resolved.is_file() or not resolved.is_relative_to((exported / 'checkpoints').resolve()):
            raise ValueError(f'Refusing to replace unmanaged ONNX symlink: {link}')
    elif link.exists():
        if not latest_manifest.is_file() or json.loads(latest_manifest.read_text()).get('onnx_sha256') != sha256(link):
            raise ValueError(f'Refusing to overwrite unmanaged ONNX file: {link}')
    tmp_link = exported / f'.policy-{os.getpid()}.onnx'
    tmp_manifest = exported / f'.manifest-{os.getpid()}.json'
    try:
        shutil.copyfile(final, tmp_link)
        tmp_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(tmp_link, link)
        os.replace(tmp_manifest, latest_manifest)
    finally:
        tmp_link.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)
    return final
