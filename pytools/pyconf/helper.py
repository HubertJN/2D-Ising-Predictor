import os
import ctypes
import json
from typing import Any, Dict, List
from warnings import warn
import copy
from pathlib import Path
import logging 

logging.basicConfig(level=logging.DEBUG, filename='config.log', filemode='w', format='%(filename)s - %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cli_cache(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"Input Prompt: {*args, *kwargs}")
        user_input = func(*args, **kwargs)
        if user_input is not None:
            logger.debug(f"User Input: {user_input}")
            return user_input
    return wrapper

if os.environ.get('running_in_pytest', 'False') == 'False':
    input = cli_cache(input)
    print = cli_cache(print)

#===============================================================================
# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549

# One of the following libraries must be available to load
libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
for libname in libnames:
    try:
        cuda = ctypes.CDLL(libname)
    except OSError:
        continue
    else:
        break
else:
    raise ImportError(f'Could not load any of: {", ".join(libnames)}')

# Constants from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

# Conversions from semantic version numbers
# Borrowed from original gist and updated from the "GPUs supported" section of this Wikipedia article
# https://en.wikipedia.org/wiki/CUDA
SEMVER_TO_CORES = {
    (1, 0): 8,    # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,   # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,   # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,   # Volta
    (7, 2): 64,
    (7, 5): 64,   # Turing
    (8, 0): 64,   # Ampere
    (8, 6): 64,
}
SEMVER_TO_ARCH = {
    (1, 0): 'tesla',
    (1, 1): 'tesla',
    (1, 2): 'tesla',
    (1, 3): 'tesla',

    (2, 0): 'fermi',
    (2, 1): 'fermi',

    (3, 0): 'kepler',
    (3, 2): 'kepler',
    (3, 5): 'kepler',
    (3, 7): 'kepler',

    (5, 0): 'maxwell',
    (5, 2): 'maxwell',
    (5, 3): 'maxwell',

    (6, 0): 'pascal',
    (6, 1): 'pascal',
    (6, 2): 'pascal',

    (7, 0): 'volta',
    (7, 2): 'volta',

    (7, 5): 'turing',

    (8, 0): 'ampere',
    (8, 6): 'ampere',
}


def get_cuda_device_specs() -> List[Dict[str, Any]]:
    """Generate spec for each GPU device with format

    {
        'name': str,
        'compute_capability': (major: int, minor: int),
        'cores': int,
        'cuda_cores': int,
        'concurrent_threads': int,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float
    }
    """

    # Type-binding definitions for ctypes
    num_gpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    free_mem = ctypes.c_size_t()
    total_mem = ctypes.c_size_t()
    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    # Check expected initialization
    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(f'cuInit failed with error code {result}: {error_str.value.decode()}')
    result = cuda.cuDeviceGetCount(ctypes.byref(num_gpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(f'cuDeviceGetCount failed with error code {result}: {error_str.value.decode()}')

    # Iterate through available devices
    device_specs = []
    for i in range(num_gpus.value):
        spec = {}
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError(f'cuDeviceGet failed with error code {result}: {error_str.value.decode()}')

        # Parse specs for each device
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            spec.update(name=name.split(b'\0', 1)[0].decode())
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            spec.update(compute_capability=(cc_major.value, cc_minor.value))
            spec.update(architecture=SEMVER_TO_ARCH.get((cc_major.value, cc_minor.value), 'unknown'))
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            spec.update(
                cores=cores.value,
                cuda_cores=cores.value * SEMVER_TO_CORES.get((cc_major.value, cc_minor.value), 'unknown'))
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                spec.update(concurrent_threads=cores.value * threads_per_core.value)
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
            spec.update(gpu_clock_mhz=clockrate.value / 1000.)
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
            spec.update(mem_clock_mhz=clockrate.value / 1000.)

        # Attempt to determine available vs. free memory
        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            warn(f'cuCtxCreate failed with error code {result}: {error_str.value.decode()}')
        else:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(free_mem), ctypes.byref(total_mem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
            if result == CUDA_SUCCESS:
                spec.update(
                    total_mem_mb=total_mem.value / 1024**2,
                    free_mem_mb=free_mem.value / 1024**2)
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                warn(f'cuMemGetInfo failed with error code {result}: {error_str.value.decode()}')
            cuda.cuCtxDetach(context)
        device_specs.append(spec)
    return device_specs
#===============================================================================

#===============================================================================
# Calculate GPU Fit
def get_unused_gpu(gpu_specs, usage_specs):
    unused_gpu = {}
    for key in usage_specs.keys():
        unused_gpu[key] = gpu_specs[key] - usage_specs[key]
    return unused_gpu

def maximise_models_per_gpu(total_spec, gpu_spec):
    # This must be kept up to date with potential optimisation parameters
    # Try except blocks allow for the use of either free memory or total memory
    try:
        memory_limit = compare_memory(total_spec['memory'], gpu_spec['free_mem_b'])
    except KeyError:
        memory_limit = compare_memory(total_spec['memory'], gpu_spec['memory'])
    try:
        multiprocessor_limit = compare_multiprocessors(total_spec['cores'], gpu_spec['cuda_cores'])
    except KeyError:
        multiprocessor_limit = compare_multiprocessors(total_spec['cores'], gpu_spec['cores'])
    # The limit is the minimum of the two
    limit = min(memory_limit, multiprocessor_limit)
    return limit, memory_limit, multiprocessor_limit

def get_one_model_percent(total_spec, gpu_spec):
    limit = maximise_models_per_gpu(total_spec, gpu_spec)
    return (1/limit) * 100

def percent_to_replications(percent, total_spec, gpu_spec):
    limit = maximise_models_per_gpu(total_spec, gpu_spec)
    return int(limit * (percent/100))

def mark_gpu_use(replications, total_spec, usage_spec):
    memory = total_spec['memory'] * replications
    multiprocessors = total_spec['multiprocessors'] * replications

    usage_spec['memory'] += memory
    usage_spec['multiprocessors'] += multiprocessors

def compare_memory(model_mem, gpu_mem):
    gpu_mem_bytes = gpu_mem
    max_models = gpu_mem_bytes // model_mem
    return max_models

def compare_multiprocessors(model_mp, gpu_mp):
    max_models = gpu_mp // model_mp
    return max_models

def compare_threads_per_block(model_tpb, gpu_tpb):
    pass    

def compare_shared_memory(model_sm, gpu_sm):
    pass


#===============================================================================

#===============================================================================
# Classes that define the simulation type, used to define simulation specific details
from model_types import Simulation, ModelTypes


class SimulationSet():
    
    def __init__(self):
        self.model_types = ModelTypes
        self.models = {}

    def add_model(self, model_type, model_name):
        if model_type in self.model_types.keys():
            if model_name in self.models.keys():
                print(f"Model {model_name} already exists.")
                input =  input("Do you want to overwrite the model? (y/n)")
                if input == "n":
                    input = input("Do you want to create a new model? (y/n)")
                    if input == "y":
                        model_name = input("Please enter a new name for the model: ")
                    else:
                        print("Model not created.")
                        return
            self.models[model_name] = self.model_types[model_type](model_name)
        else:
            print(f"Model {model_type} not found.")
    
    def duplicate_model(self, model_name, set_name, new_model_name):
        
        if set_name not in self.models.keys():
            self.models[set_name] = {}

        if model_name in self.models.keys():
            if new_model_name in self.models.keys():
                print(f"Model {new_model_name} already exists.")
                print("Model not created.")
                return
            
            self.models[set_name][new_model_name] = copy.deepcopy(self.models[model_name])
        else:
            print(f"Model {model_name} not found.")

    def write_config(self):
        print("Writing config file... first name, then path to folder.")
        filename = input("Please enter a filename: ")
        if filename[-4:] != ".dat":
            filename += ".dat"
        
        config_dir = input("Please enter a path input y for default: ")
        if config_dir == "y":
            # TODO: Make this assert that the default path exists
            config_dir = Path("./configurations/")
        else:
            config_dir = Path(config_dir)
        
        full_path = config_dir / filename

        with open(full_path, 'w') as f:
            for model in self.models.values():
                f.write('## New model ##\n')
                for key, value in model.model_config.items():
                    f.write(f'{key}={value}\n')
                f.write('\n')
        pass

#===============================================================================


if __name__ == '__main__':
    # Running this script directly will print the device specs as a test of this
    print(json.dumps(get_cuda_device_specs(), indent=2))