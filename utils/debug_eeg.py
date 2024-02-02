from pylsl import StreamInlet, resolve_byprop
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QApplication
import numpy as np
import sys

# initialize the streaming layer
finished = False
print("looking for an EEG stream...")
streams = resolve_byprop("name", "UN-2019.05.50", minimum=1)

#Create the stream inlet 
inlet = StreamInlet(streams[0])

#initialize the colomns of your data and your dictionary to capture the data.
columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ',
'Gyro1','Gyro2','Gyro3', 'Battery','Counter','Validation']

# Set up plot
app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="EEG Signal Realtime Plot")
win.resize(800,600)
win.setWindowTitle('EEG Signal Realtime Plot')

# Create 8 plots for 8 channels
plots = [win.addPlot(row=i, col=0, title="Channel %d" % (i+1)) for i in range(8)]
curves = [p.plot(pen=(i,8)) for i, p in enumerate(plots)]  # create a curve for each plot
data = [np.empty((0,)) for _ in range(8)]  # create an empty array for each channel

ptr = 0

def update():
    global curves, data, ptr, inlet

    # Pull chunk from inlet
    chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=10)

    if timestamps:
        # Update the data 
        for ch_idx in range(8):
            ch_data = [sample[ch_idx] for sample in chunk]
            data[ch_idx] = np.append(data[ch_idx], ch_data)

            curves[ch_idx].setData(data[ch_idx])
            if len(data[ch_idx]) > 250:  # limit data to the latest 250 samples, about 1 second
                curves[ch_idx].setData(data[ch_idx][-250:])
        ptr += 1

# Update every 4 ms to get approximately 250 samples per second
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(4)

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec_()
