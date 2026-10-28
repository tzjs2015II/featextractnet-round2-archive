import torch


def get_device(device_str='auto'):
    """
    Unified device selector: CUDA > MPS > CPU.

    Returns a torch.device and a string identifying the backend type
    ('cuda', 'mps', or 'cpu').
    """
    if device_str == 'cpu':
        return torch.device('cpu'), 'cpu'

    if device_str == 'cuda' or (device_str == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda'), 'cuda'

    if device_str == 'mps' or (device_str == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps'), 'mps'

    return torch.device('cpu'), 'cpu'


def configure_cuda_for_training():
    """
    CUDA optimizations for training (call AFTER set_seed).
    Enables cudnn benchmark for fixed-shape inputs.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()


def device_hardware_summary():
    """
    Return a dict summarizing the available hardware for reporting.
    """
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available()
    }

    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info