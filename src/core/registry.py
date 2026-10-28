"""registry.py —— results/registry.json 追加/更新（src.core.registry）。

用途
----
实验登记表（实验计划.md §3 目录结构、AGENTS §3.4）。对单条 entry 执行
读取-合并-写回：按 id（或 experiment_id）upsert——已存在则替换该条，否则
追加。**单进程使用**：无文件锁；多进程并发写同一 registry 需自行加锁。

结构
----
与既有条目（scripts/generate_acoustic_cases.py 写入的 data_generation 条目）
风格一致，并为实验条目补标准字段：

  id / experiment_id     唯一键（重复则覆盖更新）
  type                   'experiment'（默认）或 'data_generation' 等
  tool                   来源脚本名
  created_utc            写入时间
  split_version          切分版本（如 'Protocol-A'）
  protocol               dict：切分/归一化/留出协议快照
  seed / epochs / lr     训练配置
  model / model_params   模型类名与参数量
  hardware / torch       硬件与 torch 版本（runner 协议快照同源）
  metrics                指标（可引用 metrics_seed{n}.json 路径或直接内联）
  config_file            相关配置文件相对路径（可选）
"""

import datetime as _dt
import json
import os

DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'results', 'registry.json')


def _read_registry(registry_path):
    if not os.path.exists(registry_path):
        return []
    with open(registry_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def _write_registry(registry_path, entries):
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, 'w', encoding='utf-8') as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
        fh.write('\n')


def _entry_key(entry):
    key = entry.get('id') or entry.get('experiment_id')
    if not key:
        raise ValueError(
            "entry needs 'id' or 'experiment_id' for upsert")
    return key


def append_registry(entry, registry_path=DEFAULT_REGISTRY_PATH):
    """按 id upsert 一条登记到 registry_path（默认工作区 results/registry.json）。

    Args:
        entry: dict，须含 'id' 或 'experiment_id'；含该键的既有条目会被
               覆盖更新，否则追加。
        registry_path: registry json 路径（默认 src/core 锚定的
                       <workspace>/results/registry.json）。

    Returns:
        最终 registry 条数。
    """
    key = _entry_key(entry)
    if 'experiment_id' in entry and 'id' not in entry:
        entry['id'] = key
    if 'created_utc' not in entry:
        entry['created_utc'] = _dt.datetime.now(
            _dt.timezone.utc).isoformat(timespec='seconds')

    entries = _read_registry(registry_path)
    idx = next((i for i, e in enumerate(entries) if _entry_key(e) == key),
               None)
    if idx is None:
        entries.append(entry)
    else:
        entries[idx] = entry
    _write_registry(registry_path, entries)
    return len(entries)


def new_experiment_entry(experiment_id, protocol, seed, epochs, lr, model,
                         metrics, split_version=None, model_params=None,
                         hardware=None, torch_version=None, tool=None,
                         **extra):
    """构造一条标准实验 entry（字段见模块 docstring），供 append_registry 使用。

    Args:
        experiment_id: 唯一实验标识（seed 各自独立 id 时建议带 seed 后缀）。
        protocol: dict，切分/归一化/留出协议快照（与 runner 的
                  protocol_extra 同源，保持一致性）。
        seed / epochs / lr / model: 训练配置。
        metrics: dict 或字符串（内联指标，或指向 metrics_seed{n}.json）。
        split_version: 如 'Protocol-A' / 'Protocol-B'。
        model_params: 参数量 int（可空）。
        hardware: dict（device_hardware_summary() 输出；可空）。
        torch_version: str（可空）。
        tool: 来源脚本名（默认本模块名）。
        **extra: 其余字段透传。

    Returns:
        dict entry。
    """
    return {
        'experiment_id': experiment_id,
        'id': experiment_id,
        'type': 'experiment',
        'tool': tool if tool is not None else 'src.core.registry',
        'split_version': split_version,
        'protocol': protocol,
        'seed': seed,
        'epochs': epochs,
        'lr': lr,
        'model': model,
        'model_params': model_params,
        'hardware': hardware,
        'torch': torch_version,
        'metrics': metrics,
        **extra,
    }
