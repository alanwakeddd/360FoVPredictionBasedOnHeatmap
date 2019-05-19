import pickle
import numpy as np


def xyz2ThetaPhi(x, y, z):
    theta = (np.arctan2(y, x) + np.pi) * 180 / np.pi
    phi = (np.arctan2(np.sqrt(x**2 + y**2), z) + np.pi) * 180 / np.pi
    return theta, phi


def thetaPhi2HeatmapIndex(theta, phi):
    shape1 = theta.shape
    shape2 = phi.shape
    i = (np.floor(theta.flatten() % 360 / 10)).reshape(shape1).astype(int)
    j = (np.floor(phi.flatten() % 180 / 10)).reshape(shape2).astype(int)
    return i, j


def HeatmapGenerater(x, y, z):
    heatmap = np.zeros((1, 1, 36, 18)).astype(int)

    for k in np.arange(0, x.shape[1], 10):
        heatmap0 = np.zeros((1, 1, 36, 36)).astype(int)
        x0 = x[:, k: k + 10]
        y0 = y[:, k: k + 10]
        z0 = z[:, k: k + 10]
        thetas, phis = xyz2ThetaPhi(x0, y0, z0)
        ni, nj = thetaPhi2HeatmapIndex(thetas, phis)
        for i, j in zip(ni.flatten(), nj.flatten()):
            heatmap0[0, 0, i, j] += 1
        heatmap = np.vstack((heatmap, heatmap0))

    heatmap = heatmap[1:, :, :, :]
    return heatmap


def HeatmapGeneraterSingle(x, y, z):
    heatmap = np.zeros((1, 1, 36, 36)).astype(int)

    for k in np.arange(0, x.shape[1] // 20 * 20):
        heatmap0 = np.zeros((1, 1, 36, 36)).astype(int)
        x0 = x[:, k]
        y0 = y[:, k]
        z0 = z[:, k]
        thetas, phis = xyz2ThetaPhi(x0, y0, z0)
        ni, nj = thetaPhi2HeatmapIndex(thetas, phis)
        for i, j in zip(ni.flatten(), nj.flatten()):
            heatmap0[0, 0, i, j] += 1
        heatmap = np.vstack((heatmap, heatmap0))

    heatmap = heatmap[1:, :, :, :]
    return heatmap


def train_test_index_generator(heatmap, batch_size):
    length = heatmap.shape[0]
    train = np.arange(0, length, batch_size * 2)
    test = np.arange(batch_size, length, batch_size * 2)
    pad_train = (np.ones(train.shape[0]) * batch_size).astype(int).reshape(-1, 1)
    pad_test = (np.ones(test.shape[0]) * batch_size).astype(int).reshape(-1, 1)
    train = np.hstack((train.reshape(-1, 1), pad_train))
    test = np.hstack((test.reshape(-1, 1), pad_test))
    clip = np.vstack((np.array([train]), np.array([test])))
    return clip


def get_dims(heatmap):
    dims = np.array([heatmap.shape[1:]])
    return dims


def npz_output(c, d, input, trainOrTest):
    np.savez('shanghai_{}'.format(trainOrTest) + '.npz', clips = c, dims = d, input_raw_data = input)


def shanghai_factory(filename, trainOrTest):
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile, encoding='latin1')
    infile.close()
    print(list(new_dict))
    print(type(new_dict))

    heatmap = np.zeros((1, 1, 36, 36)).astype(int)
    for video in new_dict:
        heatmap0 = HeatmapGeneraterSingle(new_dict[video]['x'], new_dict[video]['y'], new_dict[video]['z'])
        heatmap = np.vstack((heatmap, heatmap0))
        print(heatmap0.shape)

    heatmap = heatmap[1:, :, :, :]

    clips = train_test_index_generator(heatmap, 10)
    print(clips)
    print(clips.shape)

    dims = get_dims(heatmap)
    print(dims)
    print(dims.shape)

    npz_output(clips, dims, heatmap, trainOrTest)


shanghai = {"dims": [[1, 36, 36]]}
print(shanghai)

length = 10
batch_size = 2
train = np.arange(0, length, 2)
test = np.arange(1, length, 2)
pad_train = (np.ones(train.shape[0]) * batch_size).astype(int).reshape(-1, 1)
pad_test = (np.ones(test.shape[0]) * batch_size).astype(int).reshape(-1, 1)
train = np.hstack((train.reshape(-1, 1), pad_train))
test = np.hstack((test.reshape(-1, 1), pad_test))
clip = np.vstack((train, test))
print(train)
print(test)

print("Shanghai factory begining...")
if 1:
    filename = '/Users/zhangbohan/Downloads/predrnn-pp-master/data_shanghai/shanghai_dataset_xyz_test.p'
if 0:
    filename = r'C:\Users\hans\Dropbox\predrnn-pp-master\data_shanghai\shanghai_dataset_xyz_test.p'
if 0:
    filename = '/Users/zhangbohan/Downloads/predrnn-pp-master/data_shanghai/shanghai_dataset_xyz_train.p'
if 0:
    filename = r'C:\Users\hans\Dropbox\predrnn-pp-master\data_shanghai\shanghai_dataset_xyz_train.p'
shanghai_factory(filename, 'test')
