"""Compose the official single-arm MJCF twice; kinematic preview only."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'vendor/agx_arm_sim/mujoco/agilex_arm/agilex_nero/nero.xml'
DEFAULT_LAYOUT = ROOT / 'configs/simulation/dual_nero.json'


def finite_vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'{name} 需要 {size} 个有限值')
    return value


def scene_xml(model_path=DEFAULT_MODEL, layout_path=DEFAULT_LAYOUT):
    model_path, layout_path = Path(model_path).resolve(), Path(layout_path)
    source = ET.parse(model_path).getroot()
    layout = json.loads(layout_path.read_text())
    result = ET.Element('mujoco', model='dual_nero_preview')
    ET.SubElement(result, 'compiler', angle='radian')
    option = source.find('option')
    result.append(copy.deepcopy(option) if option is not None else ET.Element('option'))
    defaults = ET.SubElement(result, 'default')
    assets = ET.SubElement(result, 'asset')
    world = ET.SubElement(result, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 3', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 0.1', rgba='.2 .25 .3 1')
    table = layout['table']
    pos = finite_vector(table['pos'], 3, 'table.pos')
    size = finite_vector(table['size'], 3, 'table.size')
    if np.any(size <= 0):
        raise ValueError('桌面半尺寸必须为正')
    ET.SubElement(world, 'geom', name='table', type='box', pos=' '.join(map(str, pos)),
                  size=' '.join(map(str, size)), rgba='.6 .5 .4 1')
    for side in ('left', 'right'):
        tree = copy.deepcopy(source)
        names = {e.get('name') for e in tree.iter() if e.get('name')}
        names.update(e.get('class') for e in tree.iter('default') if e.get('class'))
        references = {'name', 'class', 'childclass', 'mesh', 'material', 'texture',
                      'joint', 'joint1', 'joint2', 'body1', 'body2', 'site', 'tendon'}
        for element in tree.iter():
            for key in references.intersection(element.attrib):
                if element.get(key) in names:
                    element.set(key, f'{side}_{element.get(key)}')
        compiler = source.find('compiler')
        mesh_dir = compiler.get('meshdir', '') if compiler is not None else ''
        for mesh in tree.findall('./asset/mesh'):
            path = (model_path.parent / mesh_dir / mesh.attrib['file']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh.set('file', str(path))
        if tree.find('default') is not None:
            defaults.extend(list(tree.find('default')))
        if tree.find('asset') is not None:
            assets.extend(list(tree.find('asset')))
        body = tree.find('./worldbody/body')
        if body is None:
            raise ValueError('官方模型中未找到根 body')
        p = finite_vector(layout[side]['pos'], 3, f'{side}.pos')
        q = finite_vector(layout[side]['quat'], 4, f'{side}.quat')
        if abs(np.linalg.norm(q) - 1) > 1e-4:
            raise ValueError('安装四元数必须归一化（wxyz）')
        body.set('pos', ' '.join(map(str, p)))
        body.set('quat', ' '.join(map(str, q)))
        world.append(body)
        for tag in ('contact', 'equality', 'actuator'):
            element = tree.find(tag)
            if element is not None:
                destination = result.find(tag)
                if destination is None:
                    destination = ET.SubElement(result, tag)
                destination.extend(list(element))
    ET.SubElement(result, 'statistic', center='.25 0 1', extent='1.4')
    return ET.tostring(result, encoding='unicode'), layout


class DualNero:
    def __init__(self, model_path=DEFAULT_MODEL, layout_path=DEFAULT_LAYOUT):
        import mujoco
        self.mj = mujoco
        xml, self.layout = scene_xml(model_path, layout_path)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.joints, self.fingers = [], []
        self.ranges = []
        for side in ('left', 'right'):
            ids = [self.model.joint(f'{side}_joint{i}').id for i in range(1, 8)]
            self.joints.append([self.model.jnt_qposadr[i] for i in ids])
            self.ranges.extend(self.model.jnt_range[ids].tolist())
            self.fingers.append([self.model.jnt_qposadr[self.model.joint(f'{side}_gripper_joint{i}').id]
                                 for i in (1, 2)])
        self.width_scale = float(self.layout['gripper']['finger_per_width'])
        self.width_offset = float(self.layout['gripper']['finger_offset'])
        if not np.isfinite([self.width_scale, self.width_offset]).all() or self.width_scale <= 0:
            raise ValueError('夹爪映射错误')

    def set_state(self, value, allow_out_of_range=False):
        value = finite_vector(value, 16, 'state/action')
        violations = []
        for ordinal, side in enumerate(('left', 'right')):
            offset = ordinal * 8
            for i, address in enumerate(self.joints[ordinal]):
                joint = self.model.joint(f'{side}_joint{i + 1}')
                if not joint.range[0] - 1e-5 <= value[offset + i] <= joint.range[1] + 1e-5:
                    violations.append(f'{side}.J{i+1}={value[offset+i]:.4f}')
                self.data.qpos[address] = value[offset + i]
            finger = value[offset + 7] * self.width_scale + self.width_offset
            if not 0 <= finger <= 0.05:
                violations.append(f'{side}.width={value[offset+7]:.4f}')
            self.data.qpos[self.fingers[ordinal][0]] = finger
            self.data.qpos[self.fingers[ordinal][1]] = -finger
        if violations and not allow_out_of_range:
            raise ValueError('轨迹超出模型限位：' + ', '.join(violations))
        self.data.qvel[:] = 0
        self.mj.mj_forward(self.model, self.data)
        return violations

    def viewer(self):
        import mujoco.viewer
        return mujoco.viewer.launch_passive(self.model, self.data)
