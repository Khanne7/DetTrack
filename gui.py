import sys
from design_gui import *
from detector_tracker import *

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)                                                  # inicializa la UI
        # Ajustes de fuente
        self.comboBox_cam.currentIndexChanged.connect(self.fn_source_cam)   # entrada de camara
        self.pushButton_file.clicked.connect(self.fn_source_file)
        self.pushButton_del.clicked.connect(self.fn_del)
        # Ajustes de detector
        self.horizontalSlider_det.valueChanged.connect(self.fn_slider_det)  # slider step = 32
        self.pushButton_info.clicked.connect(self.fn_info)
        # Botones de control
        self.pushButton_run.clicked.connect(self.run)
        self.pushButton_stop.clicked.connect(self.stop)

        
    def fn_source_cam(self,index):
        if index > 0:
            self.pushButton_file.setEnabled(False)
            self.pushButton_del.setEnabled(False)
            self.pushButton_run.setEnabled(True)
            self.source = str(index-1)
            self.label_info.setText(self.comboBox_cam.currentText() + ' selected')
        else: 
            self.pushButton_file.setEnabled(True)
            self.pushButton_run.setEnabled(False)
            self.label_info.setText('Please, select a source')

    def fn_source_file(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select a file", QtCore.QDir.homePath())

        if fileName:
            self.comboBox_cam.setEnabled(False)
            self.pushButton_del.setEnabled(True)
            self.pushButton_run.setEnabled(True)
            self.source = fileName
            self.label_info.setText('Video selected: ' + fileName)
        else:
            self.comboBox_cam.setEnabled(True)
            self.pushButton_run.setEnabled(False)

    def fn_del(self):
        self.source = ''
        self.comboBox_cam.setEnabled(True)
        self.pushButton_del.setEnabled(False)
        self.pushButton_run.setEnabled(False)
        self.label_info.setText('Please, select a source')

    def fn_slider_det(self,value):
        i = int(value/32)
        self.label_reso_det.setText(str(32*i))

    def fn_info(self):
        classes_file = '/home/anaysa/Documentos/yolov3_detector/model/{}/model.names'.format(self.comboBox_model.currentText())
        classes = open(classes_file, "r")
        classes = classes.read()
        self.label_info.setText('Available classes: \n' + str(classes))
        
    def run(self):
        self.pushButton_stop.setEnabled(True)
        self.pushButton_run.setEnabled(False)

        source = self.source
        model_folder = self.comboBox_model.currentText()
        class_filter = self.lineEdit_classes.text()
        class_filter = class_filter.split()
        reso_det = self.label_reso_det.text()
        tracker = self.comboBox_tracker.currentText()
        reso_track = self.label_reso_track.text()
        confidence = float(0.5)
        nms_thesh = float(0.4)

        self.det_track = DetectorTracker(source, model_folder, class_filter, reso_det, tracker, reso_track, confidence, nms_thesh, label_info=self.label_info)
        self.det_track.start()

    def stop(self):
        self.pushButton_run.setEnabled(True)
        self.pushButton_stop.setEnabled(False)

        self.det_track.stop()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()