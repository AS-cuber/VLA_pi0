"""Bounded NPZ transport: arrays only, no pickle, with JSON metadata."""
import io
import json
import math
import zipfile

import numpy as np

VERSION = 1
ROLES = ('base', 'left_wrist', 'right_wrist')
MAX_BYTES = 8 * 1024 * 1024
PROMPT = '用右臂将黄色长方体物体放入透明盒子，然后双臂协作合上盒盖。'


def pack(arrays, metadata):
    values = dict(arrays)
    values['meta'] = np.frombuffer(json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode(), dtype=np.uint8)
    output = io.BytesIO()
    np.savez_compressed(output, **values)
    data = output.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError('消息超过 8 MiB')
    return data


def unpack(data):
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError('消息大小错误')
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= 8 or sum(v.file_size for v in members) > MAX_BYTES:
            raise ValueError('解压大小或数组数量错误')
        for member in members:
            if not member.filename.endswith('.npy') or '/' in member.filename:
                raise ValueError('只接受具名 NPY 数组')
            name = member.filename[:-4]
            if name in arrays:
                raise ValueError('数组名重复')
            stream = io.BytesIO(archive.read(member))
            version = np.lib.format.read_magic(stream)
            readers = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                raise ValueError('不支持的 NPY 版本')
            shape, fortran, dtype = readers[version](stream)
            count = math.prod(shape)
            size = count * dtype.itemsize
            if (dtype.hasobject or dtype.kind not in 'buif' or len(shape) > 4
                    or any(v < 0 for v in shape) or not 0 <= size <= MAX_BYTES):
                raise ValueError('数组类型或形状错误')
            payload = stream.read()
            if len(payload) != size:
                raise ValueError('数组长度与头部不一致')
            arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape, order='F' if fortran else 'C').copy()
    meta = arrays.pop('meta', None)
    if meta is None or meta.dtype != np.uint8 or meta.ndim != 1 or len(meta) > 16384:
        raise ValueError('缺少有效元数据')
    metadata = json.loads(meta.tobytes().decode())
    if not isinstance(metadata, dict) or metadata.get('version') != VERSION:
        raise ValueError('协议版本错误')
    return arrays, metadata


def state16(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError('state 必须为有限的 (16,) 数组')
    return value.copy()


def actions50(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (50, 16) or not np.isfinite(value).all():
        raise ValueError('actions 必须为有限的 (50,16) 数组')
    return value.copy()


def observation(arrays, metadata):
    if set(arrays) != {'state', *(f'image_{r}' for r in ROLES)}:
        raise ValueError('观测字段错误')
    images = {}
    for role in ROLES:
        value = arrays[f'image_{role}']
        if value.shape != (480, 640, 3) or value.dtype != np.uint8:
            raise ValueError(f'{role} 必须为 uint8 RGB (480,640,3)')
        images[role] = np.ascontiguousarray(value)
    prompt, request_id = metadata.get('prompt'), metadata.get('request_id')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
        raise ValueError('任务文本错误')
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
        raise ValueError('request_id 错误')
    return {'state': state16(arrays['state']), 'images': images, 'prompt': prompt}
