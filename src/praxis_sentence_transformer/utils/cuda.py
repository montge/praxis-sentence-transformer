"""
CUDA initialization and management utilities
"""

import os
import gc
import torch
import logging
import matplotlib.pyplot as plt
from ..logger import setup_logging

logger = setup_logging("cuda-utils")

def initialize_cuda() -> tuple[torch.device, bool]:
    """
    Initialize CUDA settings with proper error handling and fallback mechanisms.
    
    Returns:
        tuple[torch.device, bool]: (device, cuda_available)
            - device: torch.device object (cuda or cpu)
            - cuda_available: boolean indicating if CUDA is available
    """
    cuda_vars = {
        'CUDA_LAUNCH_BLOCKING': os.getenv('CUDA_LAUNCH_BLOCKING', '1'),
        'TORCH_USE_CUDA_DSA': os.getenv('TORCH_USE_CUDA_DSA', '1'),
        'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES', '0')
    }
    
    # Set CUDA environment variables
    for var, value in cuda_vars.items():
        if value is not None:
            os.environ[var] = value
            logger.info(f"Setting {var}={value}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            # Set memory limits using torch CUDA functions
            torch.cuda.set_per_process_memory_fraction(0.8)  # Set memory fraction to 80%
            
            # Configure CUDA settings
            torch.backends.cuda.enable_device_side_assertions = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Test CUDA functionality
            device = torch.device('cuda')
            test_tensor = torch.zeros(1).to(device)
            logger.info("CUDA initialization successful")
            
        except RuntimeError as e:
            logger.error(f"CUDA initialization failed: {str(e)}")
            logger.warning("Falling back to CPU")
            cuda_available = False
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")
    
    return device, cuda_available

def cleanup_cuda():
    """Clean up CUDA resources"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("CUDA resources cleaned up")

def cleanup_resources():
    """Clean up all resources including CUDA, garbage collection, and plots"""
    try:
        # Clear CUDA cache
        cleanup_cuda()
        
        # Run garbage collection
        gc.collect()
        
        # Close any open plots
        plt.close('all')
        
        logger.info("All resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}") 