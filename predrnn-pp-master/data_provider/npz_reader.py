import numpy as np
if 0:
    outfile = '/Users/zhangbohan/Downloads/moving-mnist-example/moving-mnist-test.npz'
if 1:
    outfile = r'C:\Users\hans\Dropbox\predrnn-pp-master\data_provider\shanghai_test.npz'
npzfile = np.load(outfile)
print(npzfile.files)
print(npzfile['clips'])
print(npzfile['clips'].shape)
print(npzfile['dims'])
print(npzfile['dims'].shape)
# print(np.max(npzfile['input_raw_data']))
print(npzfile['input_raw_data'].shape)
print(type(npzfile))

outfile = 'shanghai_test.npz'
npzfile = np.load(outfile)
print(npzfile.files)
print(npzfile['clips'])
print(npzfile['clips'].shape)
print(npzfile['dims'])
print(npzfile['dims'].shape)
print(np.max(npzfile['input_raw_data']))
print(npzfile['input_raw_data'].shape)
print(type(npzfile))