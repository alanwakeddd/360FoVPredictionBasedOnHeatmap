import pickle
import numpy as np
from data_provider import npz_builder

if 0:
    filename = '/Users/zhangbohan/Downloads/predrnn-pp-master/data_shanghai/shanghai_dataset_xyz_test.p'
if 1:
    filename = r'C:\Users\hans\Dropbox\predrnn-pp-master\data_shanghai\shanghai_dataset_xyz_test.p'

infile = open(filename, 'rb')
new_dict = pickle.load(infile, encoding='latin1')
infile.close()

print(list(new_dict))
print(type(new_dict))
print(list(new_dict['146']))
print(new_dict['146']['y'].shape)

x0 = new_dict['146']['x'][:, :10]
y0 = new_dict['146']['y'][:, :10]
z0 = new_dict['146']['z'][:, :10]
print(x0.shape)
print(y0.shape)
print(z0.shape)

thetas, phis = npz_builder.xyz2ThetaPhi(x0, y0, z0)

heatmap0 = np.zeros((1, 36, 18)).astype(int)

for i, (theta, phi) in enumerate(zip(thetas, phis)):
    print(i)
    print(theta)
    print(phi)

ni, nj = npz_builder.thetaPhi2HeatmapIndex(thetas, phis)

print(ni)
print(nj)
for i, j in zip(ni.flatten(), nj.flatten()):
    heatmap0[0, i, j] += 1

print(heatmap0)
heatmap0 = np.array([heatmap0])
print(np.vstack((heatmap0, heatmap0)).shape)

heatmap = npz_builder.HeatmapGenerater(new_dict['146']['x'], new_dict['146']['y'], new_dict['146']['z'])
print(heatmap.shape)

clips = npz_builder.train_test_index_generator(heatmap, 1)
print(clips)
print(clips.shape)

dims = npz_builder.get_dims(heatmap)
print(dims)
print(dims.shape)

npz_builder.npz_output(clips, dims, heatmap, 'test')