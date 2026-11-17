#!/usr/bin/env python3
"""整理 vla-wu：默认列出操作，--apply 执行归类和已知临时文件清理。"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile


MODEL_FILES = (
    'nero_pi0_adapter.py', 'nero_pi0_config.py', 'prepare_nero_pi0.py',
    'run_nero_pi0.py', 'check_nero_checkpoint.py', 'diagnose_nero_actions.py',
)
PURPOSES = {
    'nero_pi0_adapter.py': '三相机与 16 维状态/动作适配公共模块；不单独运行',
    'nero_pi0_config.py': 'NERO 数据变换和 LoRA 训练配置；由运行入口读取',
    'prepare_nero_pi0.py': '连接训练集并计算归一化统计',
    'run_nero_pi0.py': '基础模型加载验证和独立微调实验',
    'check_nero_checkpoint.py': '训练检查点重载、验证集评估和基础模型对照',
    'diagnose_nero_actions.py': '读取现有评估结果，分析预测距离与夹爪误差',
    'organize_vla_wu.py': '项目文件归类与临时产物清理',
}
OBSOLETE = ('check_nero_openpi.py', 'pi0_base_smoke.py', 'pi0_nero_smoke.py', 'pi0_aloha_smoke.py')
MODEL_INIT = '''"""NERO model tools."""
from pathlib import Path
import sys

_model_dir = str(Path(__file__).resolve().parent)
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def digest(path):
    with path.open('rb') as stream:
        return stream_digest(stream)


def stream_digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def archive_matches(archive, extracted):
    """Delete an input tar only when every regular member exists with identical bytes."""
    try:
        count = 0
        with tarfile.open(archive, 'r:*') as stream:
            for member in stream:
                parts = PurePosixPath(member.name)
                if parts.is_absolute() or '..' in parts.parts:
                    return False
                target = (extracted / member.name).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    return False
                if member.isdir():
                    if not target.is_dir():
                        return False
                    continue
                if not member.isfile() or not target.is_file() or target.stat().st_size != member.size:
                    return False
                content = stream.extractfile(member)
                if content is None or stream_digest(content) != digest(target):
                    return False
                count += 1
        return count > 0
    except (OSError, tarfile.TarError):
        return False


def active_model_processes(project):
    names = {Path(name).stem for name in (*MODEL_FILES, *OBSOLETE)}
    matches = []
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name) == __import__('os').getpid():
            continue
        try:
            args = (process / 'cmdline').read_bytes().split(b'\0')
            decoded = [arg.decode(errors='replace') for arg in args if arg]
            if not decoded or 'python' not in Path(decoded[0]).name:
                continue
            cwd = (process / 'cwd').resolve()
            related = any(str(project) in arg for arg in decoded) or cwd.is_relative_to(project)
            if related and any(any(name in arg for name in names) for arg in decoded[1:]):
                matches.append({'pid': int(process.name), 'command': ' '.join(decoded)})
        except (OSError, ValueError):
            continue
    return matches


def build_plan(project):
    require(project.is_dir(), f'项目目录不存在：{project}')
    require((project / 'vendor/openpi').is_dir(), '未找到 vendor/openpi，请确认 --project')
    moves, deletions, notes = [], [], []
    for name in MODEL_FILES:
        source, target = project / name, project / 'scripts/model' / name
        if source.exists():
            require(source.is_file() and not source.is_symlink(), f'脚本不是普通文件：{source}')
            require(target.resolve().is_relative_to(project), f'目标路径越界：{target}')
            ast.parse(source.read_text(encoding='utf-8'))
            require(not target.exists() and not target.is_symlink(), f'源和目标同时存在，请先核对：{source} -> {target}')
            moves.append({'source': str(source.relative_to(project)), 'target': str(target.relative_to(project)),
                          'sha256': digest(source)})
    results = project / 'results/pi0_nero_box_v1'
    reports = [(path, read_json(path)) for path in results.glob('*/report.json')]
    full = any(data.get('status') == 'paired_evaluation_passed'
               and data.get('evaluation_scope') == 'all_validation_frames' for _, data in reports)
    if full:
        for name in OBSOLETE:
            for path in (project / name, project / 'scripts/model' / name):
                if path.is_file() and not path.is_symlink():
                    deletions.append((path, '已由完整模型评估替代的一次性脚本'))
        for name in ('pi0_base_smoke_actions.npy', 'pi0_nero_actions.npy', 'setup.log'):
            for path in (project / name, project / 'results' / name):
                if path.is_file() and not path.is_symlink():
                    deletions.append((path, '早期试跑或安装临时产物'))
    for path, data in reports:
        status = data.get('status')
        keep_failed = False
        failed_checkpoint = None
        if status == 'failed_or_interrupted' and data.get('checkpoint_dir'):
            checkpoint = Path(data['checkpoint_dir']).resolve()
            model_root = (project / 'checkpoints/pi0_nero_box_v1').resolve()
            if checkpoint.is_relative_to(model_root) and len(checkpoint.relative_to(model_root).parts) == 1:
                valid = checkpoint.is_dir() and any((p / 'params/_METADATA').is_file()
                        for p in checkpoint.iterdir() if p.is_dir() and p.name.isdecimal())
                if valid:
                    keep_failed = True
                    notes.append(f'保留含有效检查点的中断实验日志：{path.parent}')
                elif checkpoint.is_dir():
                    failed_checkpoint = checkpoint
        if status == 'failed_or_interrupted' and any(
                read_json(p).get('status') == 'validation_evaluation_passed'
                for p in path.parent.glob('*/report.json')):
            keep_failed = True
            notes.append(f'保留含有效子评估结果的运行：{path.parent}')
        obsolete = full and (path.parent.name.startswith('verify_') or
                   data.get('evaluation_scope') == 'six_anchor_smoke' or
                   status == 'trained_checkpoint_reloaded_validation_smoke_passed')
        if (status == 'failed_or_interrupted' and not keep_failed) or obsolete:
            deletions.append((path.parent, '失败运行' if status == 'failed_or_interrupted' else '已由全量对照覆盖的抽样运行'))
        if failed_checkpoint is not None:
            deletions.append((failed_checkpoint, '失败实验且没有可加载的数字编号检查点'))
    for directory in (project / '__pycache__',):
        if directory.is_dir():
            deletions.append((directory, 'Python 缓存'))
    if (project / 'scripts').is_dir():
        deletions.extend((path, 'Python 缓存') for path in (project / 'scripts').rglob('__pycache__') if path.is_dir())
    source_root = project / 'data/nero_input_v1'
    export = read_json(project / 'data/nero_lerobot_v1/export_report.json')
    if export.get('status') == 'exported_and_reload_checked_not_yet_openpi_configured' and source_root.is_dir():
        for archive in (project / 'data/nero_training_input_v1.tar.gz', project / 'nero_training_input_v1.tar.gz'):
            if archive.is_file():
                print(f'核对已展开压缩包：{archive}', flush=True)
                if archive_matches(archive, source_root):
                    deletions.append((archive, '所有归档文件均与已展开来源相同'))
                else:
                    notes.append(f'压缩包与展开目录未完全匹配，保留：{archive}')
    selected = []
    for path, reason in sorted(deletions, key=lambda item: len(item[0].parts)):
        require(path.resolve().is_relative_to(project) and path != project, f'清理路径越界：{path}')
        if not any(path.is_relative_to(parent) for parent, _ in selected):
            selected.append((path, reason))
    unknown = [p.name for p in project.glob('*.py') if p.name not in MODEL_FILES and p.name not in OBSOLETE]
    return {'moves': moves, 'deletions': [{'path': str(p.relative_to(project)), 'reason': reason} for p, reason in selected],
            'notes': notes, 'additional_root_scripts': sorted(unknown)}


def write_owned(project, relative, text, backup):
    path = project / relative
    require(path.resolve().is_relative_to(project), f'写入路径越界：{path}')
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    if path.is_file():
        saved = backup / 'original_docs' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def file_index(project):
    lines = ['# 文件索引', '', '命令均在项目根目录运行，使用 `vendor/openpi/.venv/bin/python`。', '',
             '| 文件 | 用途 | 命令或使用方式 |', '|---|---|---|']
    for path in sorted((project / 'scripts').rglob('*.py')):
        if 'vla_wu_templates' in path.parts:
            continue
        relative = path.relative_to(project).as_posix()
        purpose = PURPOSES.get(path.name, 'Python 包标记；不单独运行' if path.name == '__init__.py' else '项目工具')
        if path.name in ('nero_pi0_adapter.py', 'nero_pi0_config.py', '__init__.py'):
            command = '由其他入口导入'
        else:
            command = f'`vendor/openpi/.venv/bin/python {relative} --help`'
        lines.append(f'| `{relative}` | {purpose} | {command} |')
    lines += ['', '环境安装入口为根目录 `setup_openpi.sh`（如存在）；使用 `bash setup_openpi.sh`，已有环境日常无需运行。',
              '模型、数据、日志文件的含义见[目录说明](目录说明.md)。', '']
    return '\n'.join(lines)


def apply_plan(project, plan, templates):
    installed = project / 'scripts/tools/organize_vla_wu.py'
    unchanged = not plan['moves'] and not plan['deletions'] and installed.is_file()
    if unchanged:
        unchanged = digest(installed) == digest(Path(__file__).resolve())
        for path in templates.rglob('*.md'):
            target = project / path.relative_to(templates)
            expected = path.read_text(encoding='utf-8').replace('{{PROJECT}}', str(project))
            unchanged = unchanged and target.is_file() and target.read_text(encoding='utf-8') == expected
        index = project / 'docs/文件索引.md'
        unchanged = unchanged and index.is_file() and index.read_text(encoding='utf-8') == file_index(project)
    if unchanged:
        print(f'目录已经整理，无需重复修改：{project}')
        for note in plan['notes']:
            print(note)
        if plan['additional_root_scripts']:
            print('额外脚本保留原位：', ', '.join(plan['additional_root_scripts']))
        return
    active = active_model_processes(project)
    require(not active, '仍有项目模型进程在运行，请结束后再整理：' + json.dumps(active, ensure_ascii=False))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = project / 'logs/maintenance' / ('layout_' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    state = {'project': str(project), **plan, 'status': 'running', 'applied_moves': []}
    manifest = backup / 'layout.json'
    try:
        for item in plan['moves']:
            source, target = project / item['source'], project / item['target']
            require(digest(source) == item['sha256'], f'源文件在整理期间改变：{source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            require(digest(target) == item['sha256'], f'文件内容改变：{target}')
            state['applied_moves'].append(item)
        for relative, content in (('scripts/__init__.py', ''), ('scripts/model/__init__.py', MODEL_INIT),
                                  ('scripts/tools/__init__.py', '')):
            if not (project / relative).exists():
                write_owned(project, relative, content, backup)
        destination = project / 'scripts/tools/organize_vla_wu.py'
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).resolve()
        if source != destination.resolve():
            write_owned(project, str(destination.relative_to(project)), source.read_text(encoding='utf-8'), backup)
        installed_templates = destination.parent / 'vla_wu_templates'
        if templates.resolve() != installed_templates.resolve():
            for path in templates.rglob('*'):
                if path.is_file():
                    relative = installed_templates.relative_to(project) / path.relative_to(templates)
                    write_owned(project, str(relative), path.read_text(encoding='utf-8'), backup)
        for path in templates.rglob('*.md'):
            relative = path.relative_to(templates)
            write_owned(project, str(relative), path.read_text(encoding='utf-8').replace('{{PROJECT}}', str(project)), backup)
        for item in plan['deletions']:
            path = project / item['path']
            if path.is_symlink():
                raise RuntimeError(f'清理目标变为符号链接，已停止：{path}')
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        write_owned(project, 'docs/文件索引.md', file_index(project), backup)
        write_owned(project, 'scripts/model/README.md',
            '# 模型工具\n\n各文件说明和命令见 [文件索引](../../docs/文件索引.md) 与 [运行说明](../../docs/运行说明.md)。\n', backup)
        uploaded = project / 'vla-wu-layout.tar.gz'
        if uploaded.is_file() and not uploaded.is_symlink():
            uploaded.unlink()
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='interrupted', error=str(error))
        raise
    finally:
        manifest.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'整理完成：{project}')
    print(f'入口说明：{project / "README.md"}')
    print(f'文件索引：{project / "docs/文件索引.md"}')
    print(f'目录变更记录：{manifest}')
    for note in plan['notes']:
        print(note)
    if plan['additional_root_scripts']:
        print('以下额外脚本未确认用途，保留原位：', ', '.join(plan['additional_root_scripts']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/home/clwu/vla-wu'))
    parser.add_argument('--apply', action='store_true', help='执行整理；默认仅列出操作')
    args = parser.parse_args()
    project = args.project.resolve()
    templates = Path(__file__).resolve().parent / 'vla_wu_templates'
    require((templates / 'README.md').is_file(), '文档模板不完整，请使用完整整理包')
    plan = build_plan(project)
    if args.apply:
        print(f"归类 {len(plan['moves'])} 个脚本，清理 {len(plan['deletions'])} 项临时产物。", flush=True)
        apply_plan(project, plan, templates)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, SyntaxError) as error:
        print(f'整理停止：{error}', file=sys.stderr)
        sys.exit(1)
