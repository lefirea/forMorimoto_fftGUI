from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import hilbert
from scipy import fftpack as ffp
import pyaudio as pa


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.sr = 16000
        self.chunk = int(self.sr * 0.1)
        self.update_seconds = 50
        self.audio = pa.PyAudio()
        self.stream = self.audio.open(format=pa.paInt16,
                                      channels=1,
                                      rate=self.sr,
                                      input=True,
                                      frames_per_buffer=self.chunk,
                                      input_device_index=0  # ここを変える
                                      )

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setXRange(0, 1000)
        self.graphWidget.setYRange(0, 20)

        self.stbar = self.statusBar()

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateGraph)
        self.timer.start(self.update_seconds)

        self.maxPeak = 0
        self.fpeak = 440
        self.fp = np.array([self.fpeak - 10, self.fpeak + 10])  # 通過域端周波数[Hz] どこからどこまで
        self.fs = np.array([self.fpeak - 30, self.fpeak + 30])  # 阻止域端周波数[Hz] どこ以下、どこ以上
        self.gpass = 1  # 通過域端最大損失[dB]
        self.gstop = 10

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

        sr = self.sr
        data = self.getWaveFrame()

        data = self.bandpass(data,
                             self.sr,
                             self.fp,
                             self.fs,
                             self.gpass,
                             self.gstop)

        z = hilbert(data)
        instPhase = np.unwrap(np.angle(z))
        instFreq = np.diff(instPhase) / (2 * np.pi) * self.sr

        fft = np.fft.fft(data)
        afft = np.abs(fft)
        freq = np.fft.fftfreq(len(afft), d=1.0 / sr)
        try:
            # peaks = np.where(afft > 5)[0]
            peaks = np.median(instFreq)
            self.maxPeak = max(self.maxPeak, peaks)
            # self.stbar.showMessage(f"{freq[peaks][0]:.1f}[Hz] (max:{self.maxPeak:.1f}[Hz])")
            self.stbar.showMessage(f"{peaks:.1f}[Hz] (max:{self.maxPeak:.1f}[Hz])")
        except:
            self.stbar.showMessage(f"small peaks (max:{self.maxPeak:.1f}[Hz])")

        self.stbar.setFont(QFont('Consolus', 15))

        self.graphWidget.plotItem.clear()
        x = freq
        y = afft
        self.graphWidget.plotItem.plot(x, y)

    def getWaveFrame(self):
        ret = self.stream.read(num_frames=self.chunk, exception_on_overflow=False)
        ret = np.frombuffer(ret, dtype=np.int16) / 32768.0
        return ret

    def bandpass(self, x, samplerate, fp, fs, gpass, gstop):
        fn = samplerate / 2  # ナイキスト周波数
        wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
        ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
        b, a = signal.butter(N, Wn, "band")  # フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
        return y

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_R:
            self.maxPeak = 0


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
