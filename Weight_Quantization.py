import torch
import os
import numpy as np
import pickle

def encode16(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param
    pickle.dump(custom_dict, open(fname, 'wb'))

def decode16(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param
    return custom_dict

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param
    pickle.dump(custom_dict, open(fname, 'wb'))

def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)
        custom_dict[name] = param
    return custom_dict

print(f"\noriginal cost: {os.stat('student_custom_small.bin').st_size} bytes.")
params = torch.load('student_custom_small.bin')
encode16(params, '16_bit_model.pkl')
print(f"16-bit cost: {os.stat('16_bit_model.pkl').st_size} bytes.")
encode8(params, '8_bit_model.pkl')
print(f"8-bit cost: {os.stat('8_bit_model.pkl').st_size} bytes.")