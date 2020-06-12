import substratools as tools
import os
import glob
import numpy as np

class MnistOpener(tools.Opener):

    @classmethod
    def _get_files(cls, folders):
        """Return list of X and y file given a folder location"""
        X_files, y_files = [], []
        for folder in folders:
            Xs = glob.glob(os.path.join(folder, 'x*.npy'))
            ys = glob.glob(os.path.join(folder, 'y*.npy'))

            X_files.extend(Xs)
            y_files.extend(ys)

        return X_files, y_files

    def get_X(self, folders):
        """Get X :-) """
        print('Finding features file...')
        X_files, _ = self._get_files(folders)
        print(X_files)
        print('Loading features...')
        Xs = []
        for X_file in X_files:
            Xs.append(np.load(X_file))
        Xs = np.concatenate(Xs)

        return Xs

    def get_y(self, folders):
        """Get y :-)"""
        print('Finding label file...')
        _, y_files = self._get_files(folders)

        print('Loading labels...')
        ys = []
        for y_file in y_files:
            ys.append(np.load(y_file))
        ys = np.concatenate(ys)

        return ys
    
    def save_predictions(self, y_pred, path):
        """Save prediction"""
        np.save(path, y_pred)

    def get_predictions(self, path):
        """Get predictions which were saved using the save_pred function"""
        return np.load(path)

    def fake_X(self):
        return np.random.randn(22, 28, 28).astype(np.float32)

    def fake_y(self):
        return np.random.choice(np.arange(10), size=(22)).astype(np.int)

