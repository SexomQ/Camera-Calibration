import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFrame, QMessageBox, QSlider, QFileDialog,
                             QProgressDialog, QCheckBox, QInputDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont

# ── Config: must match calibration settings ──────────────────────────────────
no_of_columns = 12
no_of_rows    = 8
square_size   = 60.0
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_W, DISPLAY_H = 640, 480


def make_objp():
    objp = np.zeros((no_of_rows * no_of_columns, 1, 3), dtype=np.float32)
    objp[:, 0, :2] = np.mgrid[0:no_of_columns, 0:no_of_rows].T.reshape(-1, 2) * square_size
    return objp


class TestGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.frames             = []
        self.totalFrames        = 0
        self.currentIdx         = 0
        self.pixmap             = None
        self.K                  = None   # intrinsic 3x3
        self.D                  = None   # distortion (4,1)
        self.undistortMaps      = None   # cached remap maps (display size)
        self.videoPath          = None   # path to loaded video
        self.originalFrameIndices = []   # original video frame index for each stored frame
        self.initUI()
        self.initEvents()
        self.tryAutoLoadCalib()

    # ── UI ────────────────────────────────────────────────────────────────────

    def initUI(self):
        self.setWindowTitle('Calibration Test & Visualization  [Fisheye]')
        self.setGeometry(20, 80, 910, 730)

        # Top bar
        self.loadCalibBtn = QPushButton('Load Calibration', self)
        self.loadCalibBtn.resize(130, 36)
        self.loadCalibBtn.move(10, 10)

        self.loadMediaBtn = QPushButton('Load Video / Image', self)
        self.loadMediaBtn.resize(140, 36)
        self.loadMediaBtn.move(150, 10)

        self.calibStatusLabel = QLabel('No calibration loaded', self)
        self.calibStatusLabel.resize(320, 36)
        self.calibStatusLabel.move(300, 10)
        self.calibStatusLabel.setStyleSheet('color: red;')

        # Image display
        self.imageLabel = QLabel('Load a video or image to begin', self)
        self.imageLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.imageLabel.resize(DISPLAY_W, DISPLAY_H)
        self.imageLabel.move(10, 55)
        self.imageLabel.setFrameShape(QFrame.Box)

        # ── Right panel ───────────────────────────────────────────────────────
        px = DISPLAY_W + 20
        pw = 910 - px - 10
        mono = QFont('Courier', 8)
        title_font = QFont('Courier', 9)

        def make_title(text, y):
            lbl = QLabel(text, self)
            lbl.move(px, y)
            lbl.resize(pw, 18)
            lbl.setFont(title_font)
            lbl.setStyleSheet('color: #d4a017;')
            return lbl

        def make_box(y, h):
            lbl = QLabel('—', self)
            lbl.move(px, y)
            lbl.resize(pw, h)
            lbl.setFont(mono)
            lbl.setWordWrap(True)
            lbl.setFrameShape(QFrame.Box)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            return lbl

        make_title('── Intrinsic  K ──', 55)
        self.intrinsicLabel = make_box(75, 72)

        make_title('── Distortion  D ──', 153)
        self.distortionLabel = make_box(172, 54)

        make_title('── Focal Length ──', 232)
        self.focalLabel = make_box(250, 40)

        make_title('── Principal Point ──', 296)
        self.ppLabel = make_box(314, 40)

        make_title('── Reprojection Error ──', 360)
        self.errorLabel = make_box(378, 40)

        make_title('── RMS (from calib) ──', 424)
        self.rmsLabel = make_box(442, 28)

        # Checkboxes
        self.undistortCheck = QCheckBox('Show undistorted frame', self)
        self.undistortCheck.move(px, 482)
        self.undistortCheck.resize(pw, 24)

        self.overlayCheck = QCheckBox('Project chessboard grid', self)
        self.overlayCheck.move(px, 508)
        self.overlayCheck.resize(pw, 24)

        # Status label
        self.statusLabel = QLabel('Load calibration and a video/image to begin.', self)
        self.statusLabel.resize(DISPLAY_W, 22)
        self.statusLabel.move(10, 538)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet('color: gray;')

        # Slider
        self.frameSlider = QSlider(Qt.Horizontal, self)
        self.frameSlider.resize(DISPLAY_W, 22)
        self.frameSlider.move(10, 563)
        self.frameSlider.setEnabled(False)

        # Navigation
        self.prevBtn = QPushButton('< Prev', self)
        self.prevBtn.resize(75, 30)
        self.prevBtn.move(10, 593)
        self.prevBtn.setEnabled(False)

        self.framePosLabel = QLabel('Frame: - / -', self)
        self.framePosLabel.resize(130, 30)
        self.framePosLabel.move(90, 593)
        self.framePosLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.nextBtn = QPushButton('Next >', self)
        self.nextBtn.resize(75, 30)
        self.nextBtn.move(225, 593)
        self.nextBtn.setEnabled(False)

        self.detectBtn = QPushButton('Detect & Reproject', self)
        self.detectBtn.resize(140, 30)
        self.detectBtn.move(308, 593)
        self.detectBtn.setEnabled(False)

        self.saveBtn = QPushButton('Save Undistorted', self)
        self.saveBtn.resize(130, 30)
        self.saveBtn.move(455, 593)
        self.saveBtn.setEnabled(False)

        self.sigLabel = QLabel('Coded by Obeida ElJundi & Mohammed Dhaybi', self)
        self.sigLabel.resize(910, 20)
        self.sigLabel.move(0, 705)
        self.sigLabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

    def initEvents(self):
        self.loadCalibBtn.clicked.connect(self.loadCalibClicked)
        self.loadMediaBtn.clicked.connect(self.loadMediaClicked)
        self.prevBtn.clicked.connect(self.prevFrame)
        self.nextBtn.clicked.connect(self.nextFrame)
        self.frameSlider.valueChanged.connect(self.sliderChanged)
        self.detectBtn.clicked.connect(self.detectAndReproject)
        self.saveBtn.clicked.connect(self.saveUndistorted)
        self.undistortCheck.stateChanged.connect(self.onUndistortToggled)
        self.overlayCheck.stateChanged.connect(lambda _: self.showFrame(self.currentIdx))

    # ── Calibration loading ───────────────────────────────────────────────────

    def tryAutoLoadCalib(self):
        kp = './output/K.npy'
        dp = './output/D.npy'
        if os.path.exists(kp) and os.path.exists(dp):
            self.K = np.load(kp)
            self.D = np.load(dp)
            self.undistortMaps = None
            self.updateParamPanel()
            self.calibStatusLabel.setText('Calibration loaded ✓  (from ./output/)')
            self.calibStatusLabel.setStyleSheet('color: green; font-weight: bold;')
            # Load RMS if available
            try:
                with open('./output/intrinsic.txt') as f:
                    for line in f:
                        if 'RMS' in line:
                            self.rmsLabel.setText(line.strip())
            except Exception:
                pass

    def loadCalibClicked(self):
        kpath, _ = QFileDialog.getOpenFileName(self, 'Open K.npy', './output', 'NumPy (*.npy)')
        if not kpath:
            return
        dpath, _ = QFileDialog.getOpenFileName(self, 'Open D.npy', './output', 'NumPy (*.npy)')
        if not dpath:
            return
        try:
            self.K = np.load(kpath)
            self.D = np.load(dpath)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to load calibration:\n{e}')
            return
        self.undistortMaps = None
        self.updateParamPanel()
        self.calibStatusLabel.setText('Calibration loaded ✓')
        self.calibStatusLabel.setStyleSheet('color: green; font-weight: bold;')

    def updateParamPanel(self):
        K, D = self.K, self.D
        def fmt_row(row):
            return '  ' + '  '.join(f'{v:10.3f}' for v in row)
        self.intrinsicLabel.setText('\n'.join(fmt_row(r) for r in K))
        d = D.flatten()
        self.distortionLabel.setText(
            f'  k1={d[0]:.6f}  k2={d[1]:.6f}\n  k3={d[2]:.6f}  k4={d[3]:.6f}')
        self.focalLabel.setText(f'  fx={K[0,0]:.2f} px\n  fy={K[1,1]:.2f} px')
        self.ppLabel.setText(f'  cx={K[0,2]:.2f} px\n  cy={K[1,2]:.2f} px')

    # ── Media loading ─────────────────────────────────────────────────────────

    def loadMediaClicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Video or Image', '',
            'Media (*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp)'
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            self.loadImage(path)
        else:
            self.loadVideo(path)

    def loadImage(self, path):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, 'Error', 'Could not read image.')
            return
        img = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            self.frames = [buf]
            self.totalFrames = 1
            self.currentIdx = 0
            self.frameSlider.setEnabled(False)
            self.prevBtn.setEnabled(False)
            self.nextBtn.setEnabled(False)
            self.detectBtn.setEnabled(True)
            self.saveBtn.setEnabled(True)
            self.videoPath = path
            self.originalFrameIndices = [0]
            self.showFrame(0)

    def loadVideo(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, 'Error', 'Could not open video.')
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step, ok = QInputDialog.getInt(
            self, 'Frame sampling',
            f'Video has {total} frames.\nLoad every N-th frame  (1 = all frames):',
            value=1, min=1, max=total
        )
        if not ok:
            cap.release()
            return

        estimated = (total + step - 1) // step
        progress = QProgressDialog(
            f'Loading every {step}-th frame…  (~{estimated} frames)',
            'Cancel', 0, total, self)
        progress.setWindowTitle('Loading')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        frames = []
        orig_indices = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                enc_ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if enc_ok:
                    frames.append(buf)
                    orig_indices.append(frame_idx)
            frame_idx += 1
            progress.setValue(frame_idx)
            QApplication.processEvents()
            if progress.wasCanceled():
                cap.release()
                progress.close()
                return
        cap.release()
        progress.close()
        self.videoPath = path
        self.originalFrameIndices = orig_indices
        self.frames = frames
        self.totalFrames = len(frames)
        self.currentIdx = 0
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(self.totalFrames - 1)
        self.frameSlider.setValue(0)
        self.frameSlider.setEnabled(True)
        self.prevBtn.setEnabled(True)
        self.nextBtn.setEnabled(True)
        self.detectBtn.setEnabled(True)
        self.saveBtn.setEnabled(True)
        self.showFrame(0)

    # ── Frame display ─────────────────────────────────────────────────────────

    def showFrame(self, index, frame=None):
        if not self.frames:
            return
        if frame is None:
            frame = cv2.imdecode(self.frames[index], cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.currentIdx = index
        if self.totalFrames > 1:
            self.framePosLabel.setText(f'Frame: {index + 1} / {self.totalFrames}')

        if self.undistortCheck.isChecked() and self.K is not None:
            frame = self.undistortFrame(frame)

        if self.overlayCheck.isChecked() and self.K is not None:
            frame = self.drawProjectedGrid(frame)

        self.pixmap = self.toPixmap(frame)
        self.update()

    def sliderChanged(self, value):
        self.statusLabel.setText('')
        self.statusLabel.setStyleSheet('color: gray;')
        self.showFrame(value)

    def prevFrame(self):
        if self.currentIdx > 0:
            self.frameSlider.blockSignals(True)
            self.frameSlider.setValue(self.currentIdx - 1)
            self.frameSlider.blockSignals(False)
            self.showFrame(self.currentIdx - 1)

    def nextFrame(self):
        if self.currentIdx < self.totalFrames - 1:
            self.frameSlider.blockSignals(True)
            self.frameSlider.setValue(self.currentIdx + 1)
            self.frameSlider.blockSignals(False)
            self.showFrame(self.currentIdx + 1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prevFrame()
        elif event.key() == Qt.Key_Right:
            self.nextFrame()

    # ── Detection & reprojection ──────────────────────────────────────────────

    def detectAndReproject(self):
        if not self.frames:
            return
        if self.K is None or self.D is None:
            QMessageBox.warning(self, 'No calibration', 'Please load a calibration first.')
            return

        frame = cv2.imdecode(self.frames[self.currentIdx], cv2.IMREAD_COLOR)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (no_of_columns, no_of_rows), cv2.CALIB_CB_FAST_CHECK)

        if not ret:
            self.statusLabel.setStyleSheet('color: red;')
            self.statusLabel.setText(
                f'No corners found. Ensure board has {no_of_columns}×{no_of_rows} inner corners.')
            self.errorLabel.setText('N/A — no corners found')
            return

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw detected corners in blue (cyan)
        for pt in corners.reshape(-1, 2):
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 200, 0), -1)

        # Estimate pose: undistort points first, then solvePnP
        objp = make_objp().reshape(-1, 1, 3)
        undist_corners = cv2.fisheye.undistortPoints(corners, self.K, self.D, P=self.K)
        retval, rvec, tvec = cv2.solvePnP(
            objp, undist_corners, self.K, np.zeros((4, 1)))

        if not retval:
            self.statusLabel.setStyleSheet('color: red;')
            self.statusLabel.setText('Pose estimation failed.')
            return

        # Project 3D points back onto image with fisheye model
        projected, _ = cv2.fisheye.projectPoints(
            objp, rvec, tvec, self.K, self.D)
        projected = projected.reshape(-1, 2)

        for pt in projected:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        # Draw connecting lines between detected and projected
        detected_pts = corners.reshape(-1, 2)
        for d, p in zip(detected_pts, projected):
            cv2.line(frame, (int(d[0]), int(d[1])), (int(p[0]), int(p[1])),
                     (0, 255, 255), 1)

        # Compute reprojection error
        errs = np.linalg.norm(detected_pts - projected, axis=1)
        mean_err = float(np.mean(errs))
        max_err  = float(np.max(errs))
        self.errorLabel.setText(f'  Mean: {mean_err:.3f} px\n  Max:  {max_err:.3f} px')

        if mean_err < 1.0:
            color = 'green'
            quality = 'Excellent'
        elif mean_err < 2.0:
            color = 'orange'
            quality = 'Acceptable'
        else:
            color = 'red'
            quality = 'Poor'

        self.statusLabel.setStyleSheet(f'color: {color}; font-weight: bold;')
        self.statusLabel.setText(
            f'[{quality}]  Cyan=detected  Red=projected  Yellow=error lines  |  '
            f'Mean error: {mean_err:.3f} px')

        self.showFrame(self.currentIdx, frame=frame)

    def onUndistortToggled(self):
        self.undistortMaps = None  # force recompute
        self.showFrame(self.currentIdx)

    def saveUndistorted(self):
        if self.K is None or self.D is None:
            QMessageBox.warning(self, 'No calibration', 'Load a calibration first.')
            return
        if not self.videoPath:
            QMessageBox.warning(self, 'No media', 'Load a video or image first.')
            return

        # Re-read the current frame at original resolution
        orig_idx = self.originalFrameIndices[self.currentIdx]
        ext = os.path.splitext(self.videoPath)[1].lower()

        if ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            full_frame = cv2.imread(self.videoPath)
        else:
            cap = cv2.VideoCapture(self.videoPath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
            ret, full_frame = cap.read()
            cap.release()
            if not ret or full_frame is None:
                QMessageBox.warning(self, 'Error', 'Could not read original frame.')
                return

        h, w = full_frame.shape[:2]

        # Scale K from calibration resolution (640x480) to original resolution
        sx = w / DISPLAY_W
        sy = h / DISPLAY_H
        K_scaled = self.K.copy()
        K_scaled[0, 0] *= sx   # fx
        K_scaled[1, 1] *= sy   # fy
        K_scaled[0, 2] *= sx   # cx
        K_scaled[1, 2] *= sy   # cy

        # Compute undistort maps at original resolution using scaled K
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_scaled, self.D, (w, h), np.eye(3), balance=1.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_scaled, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        undistorted = cv2.remap(full_frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Ask where to save
        default_name = f'undistorted_frame{orig_idx:05d}.png'
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Undistorted Frame',
            os.path.join(os.path.dirname(self.videoPath), default_name),
            'Images (*.png *.jpg *.bmp)'
        )
        if not save_path:
            return

        cv2.imwrite(save_path, undistorted)
        self.statusLabel.setStyleSheet('color: green; font-weight: bold;')
        self.statusLabel.setText(f'Saved {w}×{h} undistorted frame → {os.path.basename(save_path)}')

    # ── Undistort ─────────────────────────────────────────────────────────────

    def undistortFrame(self, frame):
        if self.undistortMaps is None:
            h, w = frame.shape[:2]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, (w, h), np.eye(3), balance=1.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
            self.undistortMaps = (map1, map2)
        return cv2.remap(frame, self.undistortMaps[0], self.undistortMaps[1],
                         interpolation=cv2.INTER_LINEAR)

    # ── Grid overlay ──────────────────────────────────────────────────────────

    def drawProjectedGrid(self, frame):
        # Use a fixed reference pose (board flat in front of camera at ~500mm)
        objp = make_objp().reshape(-1, 1, 3)
        # Rough initial rvec/tvec guess – works visually even if not exact pose
        rvec = np.array([[0.0], [np.pi], [0.0]])
        board_w = (no_of_columns - 1) * square_size
        board_h = (no_of_rows - 1) * square_size
        tvec = np.array([[-board_w / 2], [-board_h / 2], [600.0]])
        try:
            pts2d, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, self.K, self.D)
            pts2d = pts2d.reshape(-1, 2).astype(int)
        except cv2.error:
            return frame

        for row in range(no_of_rows):
            for col in range(no_of_columns - 1):
                i = row * no_of_columns + col
                p1, p2 = tuple(pts2d[i]), tuple(pts2d[i + 1])
                if self.inBounds(p1) and self.inBounds(p2):
                    cv2.line(frame, p1, p2, (0, 255, 0), 1)
        for col in range(no_of_columns):
            for row in range(no_of_rows - 1):
                i = row * no_of_columns + col
                p1, p2 = tuple(pts2d[i]), tuple(pts2d[i + no_of_columns])
                if self.inBounds(p1) and self.inBounds(p2):
                    cv2.line(frame, p1, p2, (0, 255, 0), 1)
        for pt in pts2d:
            if self.inBounds(tuple(pt)):
                cv2.circle(frame, tuple(pt), 3, (0, 200, 255), -1)
        return frame

    def inBounds(self, pt):
        return 0 <= pt[0] < DISPLAY_W and 0 <= pt[1] < DISPLAY_H

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        if self.pixmap:
            painter = QPainter(self)
            self.imageLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.imageLabel.setText('')
            painter.drawPixmap(10, 55, self.pixmap)

    def toPixmap(self, frame):
        img = QImage(frame, frame.shape[1], frame.shape[0],
                     frame.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(img.rgbSwapped())


app = QApplication(sys.argv)
window = TestGUI()
window.show()
sys.exit(app.exec_())
