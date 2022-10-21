from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fftpack as ffp
import pyaudio as pa


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.chunk = 1024
        self.fs = 16000
        self.update_seconds = 50
        self.audio = pa.PyAudio()
        self.stream = self.audio.open(format=pa.paInt16,
                                      channels=1,
                                      rate=self.fs,
                                      input=True,
                                      frames_per_buffer=self.chunk)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setXRange(0, 1000)
        self.graphWidget.setYRange(0, 20)

        self.stbar = self.statusBar()

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateGraph)
        self.timer.start(self.update_seconds)

    def updateGraph(self):
        # fs, data = wavfile.read("ddo.wav")
        # # print(fs, data.dtype, data.max())
        # if len(data.shape) == 2:
        #     data = data[:, 0]
        #
        # if data.dtype == np.int16:
        #     data = (data / 2 ** 15).astype(float)
        # elif data.dtype == np.int32:
        #     data = (data / 2 ** 31).astype(float)

        fs = self.fs
        data = self.getWaveFrame()

        fft = np.fft.fft(data)
        afft = np.abs(fft)
        freq = np.fft.fftfreq(len(afft), d=1.0 / fs)
        try:
            peaks = np.where(afft > 5)[0]
            self.stbar.showMessage(f"{freq[peaks][0]:.1f}[Hz]")
        except:
            self.stbar.showMessage(f"small peaks")

        self.stbar.setFont(QFont('Consolus', 15))

        self.graphWidget.plotItem.clear()
        x = freq
        y = afft
        self.graphWidget.plotItem.plot(x, y)

    def getWaveFrame(self):
        ret = self.stream.read(num_frames=self.chunk, exception_on_overflow=False)
        ret = np.frombuffer(ret, dtype=np.int16) / 32768.0
        return ret


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
