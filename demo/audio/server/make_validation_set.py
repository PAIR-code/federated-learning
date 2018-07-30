import glob
import numpy as np

labels = {'accio': 0, 'expelliarmus': 1, 'lumos': 2, 'nox': 3}
files = glob.glob('data/**/**/*.npy')
x = np.vstack([np.load(f) for f in files])
y = np.array([labels[f.split('/')[2]] for f in files])
np.save('val-inputs.npy', x)
np.save('val-labels.npy', y)


