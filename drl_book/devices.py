"""Explicit device selection; shared-memory training stays on CPU."""
import os
import torch


def device_for():
    requested=os.environ.get('DRL_BOOK_DEVICE','cpu').lower()
    if requested=='auto':
        requested='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available in this Python environment')
    if requested=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this Python environment')
    if requested not in ('cpu','mps','cuda'):
        raise ValueError('DRL_BOOK_DEVICE must be cpu, mps, cuda, or auto')
    return requested
