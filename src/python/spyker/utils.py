import spyker.spyker_plugin as impl


torch_avail = True
try: import torch
except: torch_avail = False


numpy_avail = True
try: import numpy as np
except: numpy_avail = False


def create_tensor(device, type, shape, pinned=False, unified=False, data=None):
    if data is None:
        return impl.create_tensor(device, type, shape, pinned, unified)
    else:
        return impl.create_tensor(data, device, type, shape, pinned, unified)


def torch2type(type):
    if type == torch.int8: return impl.i8
    if type == torch.int16: return impl.i16
    if type == torch.int32: return impl.i32
    if type == torch.int64: return impl.i64
    if type == torch.uint8: return impl.u8
    # if type == torch.uint16: return impl.u16
    # if type == torch.uint32: return impl.u32
    # if type == torch.uint64: return impl.u64
    if type == torch.float16: return impl.f16
    if type == torch.float32: return impl.f32
    if type == torch.float64: return impl.f64
    raise TypeError(f'Given PyTorch tensor data type {type} is not supported.')


def torch2device(device):
    if device.type == 'cpu': return impl.device('cpu')
    if device.type == 'cuda': return impl.device('cuda', device.index)
    raise TypeError(f'Given PyTorch tensor device {device} is not supported.')


def wrap_torch(input):
    if not input.is_contiguous():
        raise TypeError('Input array is not contiguous. use "contiguous" function to make it contiguous.')
    return create_tensor(torch2device(input.device), torch2type(input.dtype), input.shape, data=input.data_ptr())


def numpy2type(type):
    if type == np.int8: return impl.i8
    if type == np.int16: return impl.i16
    if type == np.int32: return impl.i32
    if type == np.int64: return impl.i64
    if type == np.uint8: return impl.u8
    if type == np.uint16: return impl.u16
    if type == np.uint32: return impl.u32
    if type == np.uint64: return impl.u64
    if type == np.float16: return impl.f16
    if type == np.float32: return impl.f32
    if type == np.float64: return impl.f64
    raise TypeError('Given Numpy array data type {type} is not supported.')


def wrap_numpy(input, write):
    if not input.flags['C_CONTIGUOUS']:
        raise TypeError('Input array is not contiguous. use "numpy.ascontiguousarray" function to make it contiguous.')

    if write and not input.flags['WRITEABLE']:
        raise TypeError('Input array is not writable. use "numpy.array(..., copy=True)" to make it writable.')

    ptr, _ = input.__array_interface__['data']
    return create_tensor(impl.device('cpu'), numpy2type(input.dtype), input.shape, data=ptr)


def type2torch(type):
    if type == impl.i8: return torch.int8
    if type == impl.i16: return torch.int16
    if type == impl.i32: return torch.int32
    if type == impl.i64: return torch.int64
    if type == impl.u8: return torch.uint8
    # if type == impl.u16: return torch.uint16
    # if type == impl.u32: return torch.uint32
    # if type == impl.u64: return torch.uint64
    if type == impl.f16: return torch.float16
    if type == impl.f32: return torch.float32
    if type == impl.f64: return torch.float64
    raise TypeError(f'Given data data type {type} is not supported by PyTorch.')


def device2torch(device):
    if device.kind == 'cpu': return torch.device('cpu')
    if device.kind == 'cuda': return torch.device(f'cuda:{device.index}')
    raise TypeError(f'Given device {device} is not supported by PyTorch.')


def create_torch(input, type, shape):
    return torch.zeros(shape, dtype=type2torch(type), device=input.device)


def type_numpy(type):
    if type == impl.i8: return np.int8
    if type == impl.i16: return np.int16
    if type == impl.i32: return np.int32
    if type == impl.i64: return np.int64
    if type == impl.u8: return np.uint8
    if type == impl.u16: return np.uint16
    if type == impl.u32: return np.uint32
    if type == impl.u64: return np.uint64
    if type == impl.f16: return np.float16
    if type == impl.f32: return np.float32
    if type == impl.f64: return np.float64
    raise TypeError(f'Given data type {type} is not supported by Numpy.')


def create_numpy(input, type, shape):
    return np.zeros(shape, dtype=type_numpy(type))


def wrap(input, write=False):
    if torch_avail and torch.is_tensor(input):
        return wrap_torch(input)

    if numpy_avail and type(input) is np.ndarray:
        return wrap_numpy(input, write)

    if type(input) is impl.tensor: return input

    raise TypeError(f'Input array {type(input)} can only be Numpy array or PyTorch tensor (if installed) or Spyker tensor.')


def create(input, dtype, shape):
    if torch_avail and torch.is_tensor(input):
        return create_torch(input, dtype, shape)

    if numpy_avail and type(input) is np.ndarray:
        return create_numpy(input, dtype, shape)

    if type(input) is impl.tensor:
        return create_tensor(input.device, dtype, shape)

    raise TypeError(f'Input array {type(input)} can only be Numpy array or PyTorch tensor (if installed) or Spyker tensor.')


def copy(input):
    if torch_avail and torch.is_tensor(input):
            return input.clone()

    if numpy_avail and type(input) is np.ndarray:
            return np.array(input, copy=True)

    if type(input) is impl.tensor: return input.copy()


def wrap_array(input):
    return wrap(input)


def _to_tensor(input, pinned=False, unified=False):
    """
    Create Spyker Tensor from PyTorch tensor or Numpy array.

    Parameters
    ----------
    input : torch.tensor or numpy.ndarray
        Container to be converted

    Returns
    -------
    Spyker.tensor
        Tensor created from copying input data
    """

    temp = wrap(input)
    output = create_tensor(temp.device, temp.dtype, temp.shape, pinned, unified)
    temp.to(output)
    return output


def _to_numpy(input):
    """
    Create Numpy array from Spyker Tensor

    Parameters
    ----------
    input : Spyker.tensor
        Tensor to be converted

    Returns
    -------
    numpy.ndarray
        Array created from copying input data
    """

    dtype = type_numpy(input.dtype)
    output = np.zeros(input.shape, dtype=dtype)
    input.to(wrap(output))
    return output


def _to_torch(input):
    """
    Create PyTorch tensor from Spyker Tensor

    Parameters
    ----------
    input : Spyker.tensor
        Tensor to be converted

    Returns
    -------
    torch.tensor
        Tensor created from copying input data
    """

    dtype = type2torch(input.dtype)
    device = device2torch(input.device)
    output = torch.zeros(input.shape, dtype=dtype, device=device)
    input.to(wrap(output))
    return output


def _to_sparse(input, threshold=0.0):
    """
    Create Spyker Sparse from PyTorch tensor or Numpy array.

    Parameters
    ----------
    input : torch.tensor or numpy.ndarray
        Container to be converted

    Returns
    -------
    spyker.Sparse
        Sparse container created from converting input data
    """

    return impl.sparse_tensor(wrap(input), threshold)


def to_tensor(*inputs, pinned=False, unified=False):
    output = tuple([_to_tensor(x, pinned, unified) for x in inputs])
    return output if len(output) > 1 else output[0]


def to_numpy(*inputs):
    output = tuple([_to_numpy(x) for x in inputs])
    return output if len(output) > 1 else output[0]


def to_torch(*inputs):
    output = tuple([_to_torch(x) for x in inputs])
    return output if len(output) > 1 else output[0]


def to_sparse(*inputs, threshold=0.0):
    output = tuple([_to_sparse(x, threshold) for x in inputs])
    return output if len(output) > 1 else output[0]


def least2(input):
    shape = list(input.shape)
    if len(shape) == 1: shape.insert(0, 1)
    if len(shape) <= 1: raise ValueError("Input dimensions couldn't be viewed as at least 2D.")
    return input.reshape(shape)


def least3(input):
    shape = list(input.shape)
    if len(shape) == 2: shape.insert(0, 1)
    if len(shape) <= 2: raise ValueError("Input dimensions couldn't be viewed as at least 3D.")
    return input.reshape(shape)


def to2(input):
    shape = list(input.shape)
    if len(shape) == 1: shape.insert(0, 1)
    if len(shape) != 2: raise ValueError("Input dimensions couldn't be viewed as 2D.")
    return input.reshape(shape)


def to3(input):
    shape = list(input.shape)
    if len(shape) == 2: shape.insert(0, 1)
    if len(shape) != 3: raise ValueError("Input dimensions couldn't be viewed as 3D.")
    return input.reshape(shape)


def to4(input):
    shape = list(input.shape)
    if len(shape) == 3: shape.insert(0, 1)
    if len(shape) != 4: raise ValueError("Input dimensions couldn't be viewed as 4D.")
    return input.reshape(shape)


def to5(input):
    shape = list(input.shape)
    if len(shape) == 4: shape.insert(0, 1)
    if len(shape) != 5: raise ValueError("Input dimensions couldn't be viewed as 5D.")
    return input.reshape(shape)


def expand2(shape):
    if isinstance(shape, int): return (shape, shape)
    if isinstance(shape, float) and shape.is_integer(): return (shape, shape)
    if isinstance(shape, (list, tuple)) and len(shape) == 1: return (shape[0], shape[0])
    if isinstance(shape, (list, tuple)) and len(shape) == 2: return tuple(shape)
    raise ValueError("Given shape couldn't be expanded to 2D.")


def expand4(shape):
    if isinstance(shape, int): return (shape, shape, shape, shape)
    if isinstance(shape, float) and shape.is_integer(): return (shape, shape, shape, shape)
    if isinstance(shape, (list, tuple)) and len(shape) == 1: return (shape[0], shape[0], shape[0], shape[0])
    if isinstance(shape, (list, tuple)) and len(shape) == 2: return [shape[0], shape[1], shape[0], shape[1]]
    if isinstance(shape, (list, tuple)) and len(shape) == 4: return tuple(shape)
    raise ValueError("Given shape couldn't be expanded to 4D.")


def read_mnist(data_path, label_path):
    data = impl.helper.mnist_data(data_path)
    label = impl.helper.mnist_label(label_path)
    return data, label
