import spyker.spyker_plugin as impl


torch_avail = True
try: import torch
except: torch_avail = False


numpy_avail = True
try: import numpy as np
except: numpy_avail = False


def create_tensor(device, dtype, shape, pinned=False, unified=False, data=None):
    if data is None:
        return impl.create_tensor(device, dtype, shape, pinned, unified)
    else:
        return impl.create_tensor(data, device, dtype, shape, pinned, unified)


def torch2type(dtype):
    if dtype == torch.int8: return 'i8'
    if dtype == torch.int16: return 'i16'
    if dtype == torch.int32: return 'i32'
    if dtype == torch.int64: return 'i64'
    if dtype == torch.uint8: return 'u8'
    # if dtype == torch.uint16: return 'u16'
    # if dtype == torch.uint32: return 'u32'
    # if dtype == torch.uint64: return 'u64'
    if dtype == torch.float16: return 'f16'
    if dtype == torch.float32: return 'f32'
    if dtype == torch.float64: return 'f64'
    raise TypeError(f'Given PyTorch tensor data type {dtype} is not supported.')


def torch2device(device):
    if device.type == 'cpu': return impl.device('cpu')
    if device.type == 'cuda': return impl.device('cuda', device.index)
    raise TypeError(f'Given PyTorch tensor device {device} is not supported.')


def wrap_torch(array):
    if not array.is_contiguous():
        raise TypeError('Input array is not contiguous. use "contiguous" function to make it contiguous.')
    return create_tensor(torch2device(array.device), torch2type(array.dtype), array.shape, data=array.data_ptr())


def numpy2type(dtype):
    if dtype == np.int8: return 'i8'
    if dtype == np.int16: return 'i16'
    if dtype == np.int32: return 'i32'
    if dtype == np.int64: return 'i64'
    if dtype == np.uint8: return 'u8'
    if dtype == np.uint16: return 'u16'
    if dtype == np.uint32: return 'u32'
    if dtype == np.uint64: return 'u64'
    if dtype == np.float16: return 'f16'
    if dtype == np.float32: return 'f32'
    if dtype == np.float64: return 'f64'
    raise TypeError('Given Numpy array data type {dtype} is not supported.')


def wrap_numpy(array, write):
    if not array.flags['C_CONTIGUOUS']:
        raise TypeError('Input array is not contiguous. use "numpy.ascontiguousarray" function to make it contiguous.')

    if write and not array.flags['WRITEABLE']:
        raise TypeError('Input array is not writable. use "numpy.array(..., copy=True)" to make it writable.')

    ptr, _ = array.__array_interface__['data']
    return create_tensor(impl.device('cpu'), numpy2type(array.dtype), array.shape, data=ptr)


def type2torch(dtype):
    dtype = dtype.lower()
    if dtype == 'i8': return torch.int8
    if dtype == 'i16': return torch.int16
    if dtype == 'i32': return torch.int32
    if dtype == 'i64': return torch.int64
    if dtype == 'u8': return torch.uint8
    # if dtype == 'u16': return torch.uint16
    # if dtype == 'u32': return torch.uint32
    # if dtype == 'u64': return torch.uint64
    if dtype == 'f16': return torch.float16
    if dtype == 'f32': return torch.float32
    if dtype == 'f64': return torch.float64
    raise TypeError(f'Given data data type {dtype} is not supported by PyTorch.')


def device2torch(device):
    if device.kind == 'cpu': return torch.device('cpu')
    if device.kind == 'cuda': return torch.device(f'cuda:{device.index}')
    raise TypeError(f'Given device {device} is not supported by PyTorch.')


def create_torch(array, dtype, shape):
    return torch.zeros(shape, dtype=type2torch(dtype), device=array.device)


def type_numpy(dtype):
    dtype = dtype.lower()
    if dtype == 'i8': return np.int8
    if dtype == 'i16': return np.int16
    if dtype == 'i32': return np.int32
    if dtype == 'i64': return np.int64
    if dtype == 'u8': return np.uint8
    if dtype == 'u16': return np.uint16
    if dtype == 'u32': return np.uint32
    if dtype == 'u64': return np.uint64
    if dtype == 'f16': return np.float16
    if dtype == 'f32': return np.float32
    if dtype == 'f64': return np.float64
    raise TypeError(f'Given data type {dtype} is not supported by Numpy.')


def create_numpy(array, dtype, shape):
    return np.zeros(shape, dtype=type_numpy(dtype))


def wrap(array, write=False):
    if torch_avail and torch.is_tensor(array):
        return wrap_torch(array)

    if numpy_avail and type(array) is np.ndarray:
        return wrap_numpy(array, write)

    if type(array) is impl.tensor: return array

    raise TypeError(f'Input array {type(array)} can only be Numpy array or PyTorch tensor (if installed) or Spyker tensor.')


def create(array, dtype, shape):
    if torch_avail and torch.is_tensor(array):
        return create_torch(array, dtype, shape)

    if numpy_avail and type(array) is np.ndarray:
        return create_numpy(array, dtype, shape)

    if type(array) is impl.tensor:
        return create_tensor(array.device, dtype, shape)

    raise TypeError(f'Input array {type(array)} can only be Numpy array or PyTorch tensor (if installed) or Spyker tensor.')


def copy(array):
    if torch_avail and torch.is_tensor(array):
            return array.clone()

    if numpy_avail and type(array) is np.ndarray:
            return np.array(array, copy=True)

    if type(array) is impl.tensor: return array.copy()


def wrap_array(array):
    return wrap(array)


def copy_array(source, destin):
    wrap(source, False).to(wrap(destin, True))


def _to_tensor(array, pinned=False, unified=False):
    """
    Create Spyker Tensor from PyTorch tensor or Numpy array.

    Parameters
    ----------
    array : torch.tensor or numpy.ndarray
        Container to be converted

    Returns
    -------
    Spyker.tensor
        Tensor created from copying input data
    """

    temp = wrap(array)
    output = create_tensor(temp.device, temp.dtype, temp.shape, pinned, unified)
    temp.to(output)
    return output


def _to_numpy(array):
    """
    Create Numpy array from Spyker Tensor

    Parameters
    ----------
    array : Spyker.tensor
        Tensor to be converted

    Returns
    -------
    numpy.ndarray
        Array created from copying input data
    """

    dtype = type_numpy(array.dtype)
    output = np.zeros(array.shape, dtype=dtype)
    array.to(wrap(output))
    return output


def _to_torch(array):
    """
    Create PyTorch tensor from Spyker Tensor

    Parameters
    ----------
    array : Spyker.tensor
        Tensor to be converted

    Returns
    -------
    torch.tensor
        Tensor created from copying input data
    """

    dtype = type2torch(array.dtype)
    device = device2torch(array.device)
    output = torch.zeros(array.shape, dtype=dtype, device=device)
    array.to(wrap(output))
    return output


def _to_sparse(array, threshold=0.0):
    """
    Create Spyker Sparse from PyTorch tensor or Numpy array.

    Parameters
    ----------
    array : torch.tensor or numpy.ndarray
        Container to be converted

    Returns
    -------
    spyker.Sparse
        Sparse container created from converting input data
    """

    return impl.sparse_tensor(wrap(array), threshold)


def to_tensor(*arrays, pinned=False, unified=False):
    output = tuple([_to_tensor(x, pinned, unified) for x in arrays])
    return output if len(output) > 1 else output[0]


def to_numpy(*arrays):
    output = tuple([_to_numpy(x) for x in arrays])
    return output if len(output) > 1 else output[0]


def to_torch(*arrays):
    output = tuple([_to_torch(x) for x in arrays])
    return output if len(output) > 1 else output[0]


def to_sparse(*arrays, threshold=0.0):
    output = tuple([_to_sparse(x, threshold) for x in arrays])
    return output if len(output) > 1 else output[0]


def least2(array):
    shape = list(array.shape)
    if len(shape) == 1: shape.insert(0, 1)
    if len(shape) <= 1: raise ValueError("Input dimensions couldn't be viewed as at least 2D.")
    return array.reshape(shape)


def least3(array):
    shape = list(array.shape)
    if len(shape) == 2: shape.insert(0, 1)
    if len(shape) <= 2: raise ValueError("Input dimensions couldn't be viewed as at least 3D.")
    return array.reshape(shape)


def to2(array):
    shape = list(array.shape)
    if len(shape) == 1: shape.insert(0, 1)
    if len(shape) != 2: raise ValueError("Input dimensions couldn't be viewed as 2D.")
    return array.reshape(shape)


def to3(array):
    shape = list(array.shape)
    if len(shape) == 2: shape.insert(0, 1)
    if len(shape) != 3: raise ValueError("Input dimensions couldn't be viewed as 3D.")
    return array.reshape(shape)


def to4(array):
    shape = list(array.shape)
    if len(shape) == 3: shape.insert(0, 1)
    if len(shape) != 4: raise ValueError("Input dimensions couldn't be viewed as 4D.")
    return array.reshape(shape)


def to5(array):
    shape = list(array.shape)
    if len(shape) == 4: shape.insert(0, 1)
    if len(shape) != 5: raise ValueError("Input dimensions couldn't be viewed as 5D.")
    return array.reshape(shape)


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


def read_mnist(train_images, train_labels, test_images, test_labels):
    TRX = impl.helper.mnist_data(train_images)
    TRY = impl.helper.mnist_label(train_labels)
    TEX = impl.helper.mnist_data(test_images)
    TEY = impl.helper.mnist_label(test_labels)
    return TRX, TRY, TEX, TEY
