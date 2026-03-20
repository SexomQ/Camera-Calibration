import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFrame, QCheckBox, QMessageBox, QSlider, QFileDialog,
                             QProgressDialog, QInputDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter
import os

# Config Variables - Enter their values according to your Checkerboard
no_of_columns = 12   # number of inner corners (columns)
no_of_rows    = 8    # number of inner corners (rows)
square_size   = 60.0 # size of square in mm


def make_objp():
    """Standard chessboard object points shaped (N,1,3) for cv2.fisheye."""
    objp = np.zeros((no_of_rows * no_of_columns, 1, 3), dtype=np.float32)
    objp[:, 0, :2] = np.mgrid[0:no_of_columns, 0:no_of_rows].T.reshape(-1, 2) * square_size
    return objp


class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        self.cap              = None
        self.frames           = []
        self.totalFrames      = 0
        self.currentFrameIndex = 0
        self.pixmap           = None
        self.detectingCorners = False
        self.currentCorners   = None

        # Collected calibration data
        self.confirmedImagesCounter = 0
        self.imgpoints = []   # list of (N,1,2) float32 arrays

        self.initUI()
        self.initEvents()

    # ── UI ────────────────────────────────────────────────────────────────────

    def initUI(self):
        self.setWindowTitle('Camera Calibration  [Fisheye]')
        self.setGeometry(20, 80, 660, 730)

        self.loadButton = QPushButton('Load Video', self)
        self.loadButton.resize(110, 40)
        self.loadButton.move(10, 10)

        self.detectCornersCheckbox = QCheckBox("Auto-detect corners", self)
        self.detectCornersCheckbox.move(130, 20)

        self.videoInfoLabel = QLabel('No video loaded', self)
        self.videoInfoLabel.resize(200, 20)
        self.videoInfoLabel.move(290, 20)

        self.counterLabel = QLabel('Frames confirmed: 0', self)
        self.counterLabel.resize(160, 20)
        self.counterLabel.move(490, 20)

        self.imageLabel = QLabel('Load a video to begin', self)
        self.imageLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.imageLabel.resize(1280, 720)
        self.imageLabel.move(10, 60)
        self.imageLabel.setFrameShape(QFrame.Box)

        self.statusLabel = QLabel('Navigate to a frame and click "Detect Frame" or enable Auto-detect', self)
        self.statusLabel.resize(1280, 22)
        self.statusLabel.move(10, 543)
        self.statusLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.statusLabel.setStyleSheet('color: gray;')

        self.frameSlider = QSlider(Qt.Horizontal, self)
        self.frameSlider.resize(1280, 22)
        self.frameSlider.move(10, 568)
        self.frameSlider.setEnabled(False)

        self.prevButton = QPushButton('< Prev', self)
        self.prevButton.resize(75, 30)
        self.prevButton.move(10, 598)
        self.prevButton.setEnabled(False)

        self.framePosLabel = QLabel('Frame: - / -', self)
        self.framePosLabel.resize(130, 30)
        self.framePosLabel.move(90, 598)
        self.framePosLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.nextButton = QPushButton('Next >', self)
        self.nextButton.resize(75, 30)
        self.nextButton.move(225, 598)
        self.nextButton.setEnabled(False)

        self.detectButton = QPushButton('Detect Frame', self)
        self.detectButton.resize(100, 30)
        self.detectButton.move(308, 598)
        self.detectButton.setEnabled(False)

        self.confirmButton = QPushButton('CONFIRM', self)
        self.confirmButton.resize(85, 30)
        self.confirmButton.move(415, 598)
        self.confirmButton.hide()

        self.ignoreButton = QPushButton('IGNORE', self)
        self.ignoreButton.resize(75, 30)
        self.ignoreButton.move(505, 598)
        self.ignoreButton.hide()

        self.doneButton = QPushButton('DONE', self)
        self.doneButton.resize(70, 30)
        self.doneButton.move(582, 598)

    def initEvents(self):
        self.loadButton.clicked.connect(self.loadVideoClicked)
        self.prevButton.clicked.connect(self.prevFrame)
        self.nextButton.clicked.connect(self.nextFrame)
        self.frameSlider.valueChanged.connect(self.sliderChanged)
        self.detectButton.clicked.connect(self.detectCurrentFrame)
        self.confirmButton.clicked.connect(self.confirmClicked)
        self.ignoreButton.clicked.connect(self.ignoreClicked)
        self.doneButton.clicked.connect(self.doneClicked)
        self.detectCornersCheckbox.stateChanged.connect(self.detectCornersCheckboxClicked)

    # ── Video loading ─────────────────────────────────────────────────────────

    def loadVideoClicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Video', '',
            'Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.MP4 *.AVI)'
        )
        if not path:
            return
        camId, ok = QInputDialog.getText(self, 'Camera Number', 'Camera ID (e.g. cam1, cam8):',
                                         text=getattr(self, 'camId', ''))
        if not ok:
            return
        self.camId = camId.strip() if camId.strip() else 'cam'
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, 'Error', 'Could not open video file.')
            return

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = QProgressDialog('Loading video frames...', 'Cancel', 0, total, self)
        progress.setWindowTitle('Loading')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        self.origW = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.origH = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self.frames.append(buf)
            progress.setValue(len(self.frames))
            QApplication.processEvents()
            if progress.wasCanceled():
                self.frames = []
                self.cap.release()
                self.cap = None
                progress.close()
                return

        progress.close()
        self.totalFrames = len(self.frames)
        self.currentFrameIndex = 0
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(self.totalFrames - 1)
        self.frameSlider.setValue(0)
        self.frameSlider.setEnabled(True)
        self.prevButton.setEnabled(True)
        self.nextButton.setEnabled(True)
        self.detectButton.setEnabled(True)
        self.videoInfoLabel.setText(os.path.basename(path))
        self.showFrame(0)

    # ── Frame display & navigation ────────────────────────────────────────────

    def showFrame(self, index, frame=None):
        if not self.frames:
            return
        if frame is None:
            frame = cv2.imdecode(self.frames[index], cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.currentFrameIndex = index
        self.framePosLabel.setText(f'Frame: {index + 1} / {self.totalFrames}')
        self.currentCorners = None
        self.toggleConfirmIgnore(False)

        if self.detectingCorners:
            found, corners, frame = self.detectCorners(frame)
            if found:
                self.currentCorners = corners
                self.statusLabel.setStyleSheet('color: green; font-weight: bold;')
                self.statusLabel.setText('Corners detected! Click CONFIRM to use this frame.')
                self.toggleConfirmIgnore(True)
            else:
                self.statusLabel.setStyleSheet('color: red;')
                self.statusLabel.setText('No corners found on this frame. Try another frame.')
        else:
            self.statusLabel.setStyleSheet('color: gray;')
            self.statusLabel.setText('Navigate to a frame and click "Detect Frame" or enable Auto-detect.')

        self.pixmap = self.toPixmap(frame)
        self.update()

    def detectCurrentFrame(self):
        if not self.frames:
            return
        frame = cv2.imdecode(self.frames[self.currentFrameIndex], cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.currentCorners = None
        found, corners, frame = self.detectCorners(frame)
        if found:
            self.currentCorners = corners
            self.statusLabel.setStyleSheet('color: green; font-weight: bold;')
            self.statusLabel.setText('Corners detected! Click CONFIRM to use this frame.')
            self.toggleConfirmIgnore(True)
        else:
            self.statusLabel.setStyleSheet('color: red;')
            self.statusLabel.setText(
                f'No corners found. Ensure board has {no_of_columns}×{no_of_rows} inner corners and is fully visible.'
            )
            self.toggleConfirmIgnore(False)
        self.pixmap = self.toPixmap(frame)
        self.update()

    def sliderChanged(self, value):
        self.showFrame(value)

    def prevFrame(self):
        if self.currentFrameIndex > 0:
            self.frameSlider.blockSignals(True)
            self.frameSlider.setValue(self.currentFrameIndex - 1)
            self.frameSlider.blockSignals(False)
            self.showFrame(self.currentFrameIndex - 1)

    def nextFrame(self):
        if self.currentFrameIndex < self.totalFrames - 1:
            self.frameSlider.blockSignals(True)
            self.frameSlider.setValue(self.currentFrameIndex + 1)
            self.frameSlider.blockSignals(False)
            self.showFrame(self.currentFrameIndex + 1)

    def detectCornersCheckboxClicked(self, state):
        self.detectingCorners = state == Qt.Checked
        if self.frames:
            self.showFrame(self.currentFrameIndex)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prevFrame()
        elif event.key() == Qt.Key_Right:
            self.nextFrame()

    # ── Confirm / Ignore / Done ───────────────────────────────────────────────

    def confirmClicked(self):
        self.confirmedImagesCounter += 1
        self.imgpoints.append(self.currentCorners)
        self.counterLabel.setText(f'Frames confirmed: {self.confirmedImagesCounter}')
        self.statusLabel.setStyleSheet('color: blue;')
        self.statusLabel.setText(f'Frame {self.confirmedImagesCounter} saved. Navigate to the next frame.')
        self.toggleConfirmIgnore(False)

    def ignoreClicked(self):
        self.toggleConfirmIgnore(False)
        self.statusLabel.setStyleSheet('color: gray;')
        self.statusLabel.setText('Frame ignored. Continue navigating.')

    def doneClicked(self):
        if self.confirmedImagesCounter < 3:
            rem = 3 - self.confirmedImagesCounter
            QMessageBox.warning(self, 'Warning',
                f'At least 3 frames required. Please confirm {rem} more frame(s).')
            return
        self.runFisheyeCalibration()

    def toggleConfirmIgnore(self, show):
        if show:
            self.confirmButton.show()
            self.ignoreButton.show()
        else:
            self.confirmButton.hide()
            self.ignoreButton.hide()

    # ── Fisheye calibration ───────────────────────────────────────────────────

    def runFisheyeCalibration(self):
        objp       = make_objp()
        objpoints  = [objp] * self.confirmedImagesCounter
        img_size   = (1280, 720)
        K          = np.zeros((3, 3))
        D          = np.zeros((4, 1))
        flags      = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                      cv2.fisheye.CALIB_FIX_SKEW)

        self.statusLabel.setStyleSheet('color: orange;')
        self.statusLabel.setText('Running fisheye calibration…')
        QApplication.processEvents()

        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, self.imgpoints, img_size, K, D, flags=flags
            )
        except cv2.error as e:
            QMessageBox.critical(self, 'Calibration failed',
                f'cv2.fisheye.calibrate error:\n{e}\n\nTry adding more frames or make sure the board is fully visible.')
            self.statusLabel.setStyleSheet('color: red;')
            self.statusLabel.setText('Calibration failed. See error dialog.')
            return

        self.K, self.D = K, D
        self.saveResults(K, D, rvecs, tvecs, rms)

        # Show undistorted first confirmed frame as visual feedback
        first_frame = cv2.imdecode(self.frames[self.currentFrameIndex], cv2.IMREAD_COLOR)
        undistorted = self.undistortFrame(first_frame, K, D)
        side_by_side = np.hstack([first_frame, undistorted])
        side_by_side = cv2.resize(side_by_side, (1280, 720))
        cv2.putText(side_by_side, 'Original', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(side_by_side, 'Undistorted', (650, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.pixmap = self.toPixmap(side_by_side)
        self.update()

        self.statusLabel.setStyleSheet('color: green; font-weight: bold;')
        self.statusLabel.setText(
            f'Calibration done!  RMS error: {rms:.4f} px  |  Results saved to ./output/{self.camId}/')

        QMessageBox.information(self, 'Calibration Complete',
            f'Fisheye calibration finished!\n\n'
            f'RMS reprojection error: {rms:.4f} px\n\n'
            f'fx={K[0,0]:.1f}  fy={K[1,1]:.1f}\n'
            f'cx={K[0,2]:.1f}  cy={K[1,2]:.1f}\n'
            f'k1={D[0,0]:.5f}  k2={D[1,0]:.5f}\n'
            f'k3={D[2,0]:.5f}  k4={D[3,0]:.5f}\n\n'
            f'Saved to ./output/{self.camId}/')

    def undistortFrame(self, frame, K, D):
        h, w = frame.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=1.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    def saveResults(self, K, D, rvecs, tvecs, rms):
        folder = f'./output/{self.camId}'
        os.makedirs(folder, exist_ok=True)

        # Scale K from calibration resolution (1280x720) to original video resolution
        sx = self.origW / 1280
        sy = self.origH / 720
        K_orig = K.copy()
        K_orig[0, 0] *= sx  # fx
        K_orig[1, 1] *= sy  # fy
        K_orig[0, 2] *= sx  # cx
        K_orig[1, 2] *= sy  # cy

        # NumPy binary — K at original resolution, ready to use directly
        np.save(f'{folder}/K.npy', K_orig)
        np.save(f'{folder}/D.npy', D)
        np.save(f'{folder}/rvecs.npy', np.array(rvecs))
        np.save(f'{folder}/tvecs.npy', np.array(tvecs))
        # Human-readable
        with open(f'{folder}/intrinsic.txt', 'w') as f:
            f.write(
                f'Calibration resolution: 1280x720\n'
                f'Original video resolution: {self.origW}x{self.origH}\n\n'
                f'K (scaled to {self.origW}x{self.origH})=\n{K_orig}\n\n'
                f'K (calibration 1280x720)=\n{K}\n\n'
                f'D (k1,k2,k3,k4)=\n{D}\n\n'
                f'RMS reprojection error: {rms:.6f} px\n'
            )
        with open(f'{folder}/extrinsic.txt', 'w') as f:
            for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
                f.write(f'Frame {i+1}:\n  rvec={rv.T}\n  tvec={tv.T}\n')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def detectCorners(self, image):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (no_of_columns, no_of_rows),
                                                  cv2.CALIB_CB_FAST_CHECK)
        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(image, (no_of_columns, no_of_rows), corners, ret)
        return ret, corners, image

    def toPixmap(self, image):
        img = QImage(image, image.shape[1], image.shape[0],
                     image.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(img.rgbSwapped())

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            self.imageLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.imageLabel.setText('')
            painter.drawPixmap(10, 60, self.pixmap)


app = QApplication(sys.argv)
window = MyGUI()
window.show()
sys.exit(app.exec_())
