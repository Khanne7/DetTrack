# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(320, 532)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(300, 100))
        self.groupBox.setMaximumSize(QtCore.QSize(300, 90))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.groupBox)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_3.addWidget(self.line, 0, 1, 2, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.pushButton_del = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_del.setEnabled(False)
        self.pushButton_del.setMaximumSize(QtCore.QSize(35, 16777215))
        self.pushButton_del.setObjectName("pushButton_del")
        self.horizontalLayout_7.addWidget(self.pushButton_del)
        self.gridLayout_3.addLayout(self.horizontalLayout_7, 0, 2, 1, 1)
        self.comboBox_cam = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_cam.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_cam.setObjectName("comboBox_cam")
        self.comboBox_cam.addItem("")
        self.comboBox_cam.setItemText(0, "")
        self.comboBox_cam.addItem("")
        self.comboBox_cam.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_cam, 1, 0, 1, 1)
        self.pushButton_file = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_file.setObjectName("pushButton_file")
        self.gridLayout_3.addWidget(self.pushButton_file, 1, 2, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(300, 200))
        self.tabWidget.setMaximumSize(QtCore.QSize(300, 200))
        self.tabWidget.setMouseTracking(False)
        self.tabWidget.setTabletTracking(False)
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_set_det = QtWidgets.QWidget()
        self.tab_set_det.setObjectName("tab_set_det")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_set_det)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.tab_set_det)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.lineEdit_classes = QtWidgets.QLineEdit(self.tab_set_det)
        self.lineEdit_classes.setObjectName("lineEdit_classes")
        self.horizontalLayout_4.addWidget(self.lineEdit_classes)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.tab_set_det)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.comboBox_model = QtWidgets.QComboBox(self.tab_set_det)
        self.comboBox_model.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_model.setObjectName("comboBox_model")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.horizontalLayout.addWidget(self.comboBox_model)
        self.pushButton_info = QtWidgets.QPushButton(self.tab_set_det)
        self.pushButton_info.setMaximumSize(QtCore.QSize(30, 16777215))
        self.pushButton_info.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_info.setIconSize(QtCore.QSize(16, 16))
        self.pushButton_info.setObjectName("pushButton_info")
        self.horizontalLayout.addWidget(self.pushButton_info)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.tab_set_det)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.label_reso_det = QtWidgets.QLabel(self.tab_set_det)
        self.label_reso_det.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_reso_det.setObjectName("label_reso_det")
        self.horizontalLayout_5.addWidget(self.label_reso_det)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalSlider_det = QtWidgets.QSlider(self.tab_set_det)
        self.horizontalSlider_det.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.horizontalSlider_det.setAutoFillBackground(False)
        self.horizontalSlider_det.setMaximum(1280)
        self.horizontalSlider_det.setSingleStep(32)
        self.horizontalSlider_det.setPageStep(32)
        self.horizontalSlider_det.setProperty("value", 416)
        self.horizontalSlider_det.setSliderPosition(416)
        self.horizontalSlider_det.setTracking(True)
        self.horizontalSlider_det.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_det.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_det.setTickInterval(32)
        self.horizontalSlider_det.setObjectName("horizontalSlider_det")
        self.verticalLayout.addWidget(self.horizontalSlider_det)
        self.gridLayout_2.addLayout(self.verticalLayout, 4, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem1, 3, 0, 1, 1)
        self.tabWidget.addTab(self.tab_set_det, "")
        self.tab_set_track = QtWidgets.QWidget()
        self.tab_set_track.setObjectName("tab_set_track")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_set_track)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 2, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.tab_set_track)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.comboBox_tracker = QtWidgets.QComboBox(self.tab_set_track)
        self.comboBox_tracker.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox_tracker.setObjectName("comboBox_tracker")
        self.comboBox_tracker.addItem("")
        self.comboBox_tracker.addItem("")
        self.comboBox_tracker.addItem("")
        self.comboBox_tracker.addItem("")
        self.comboBox_tracker.addItem("")
        self.comboBox_tracker.addItem("")
        self.comboBox_tracker.addItem("")
        self.horizontalLayout_3.addWidget(self.comboBox_tracker)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.tab_set_track)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.label_reso_track = QtWidgets.QLabel(self.tab_set_track)
        self.label_reso_track.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_reso_track.setObjectName("label_reso_track")
        self.horizontalLayout_6.addWidget(self.label_reso_track)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalSlider_track = QtWidgets.QSlider(self.tab_set_track)
        self.horizontalSlider_track.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.horizontalSlider_track.setAutoFillBackground(False)
        self.horizontalSlider_track.setMaximum(1280)
        self.horizontalSlider_track.setSingleStep(20)
        self.horizontalSlider_track.setPageStep(20)
        self.horizontalSlider_track.setProperty("value", 160)
        self.horizontalSlider_track.setSliderPosition(160)
        self.horizontalSlider_track.setTracking(True)
        self.horizontalSlider_track.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_track.setInvertedAppearance(False)
        self.horizontalSlider_track.setInvertedControls(False)
        self.horizontalSlider_track.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_track.setTickInterval(0)
        self.horizontalSlider_track.setObjectName("horizontalSlider_track")
        self.verticalLayout_2.addWidget(self.horizontalSlider_track)
        self.gridLayout.addLayout(self.verticalLayout_2, 3, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem4, 4, 0, 1, 1)
        self.tabWidget.addTab(self.tab_set_track, "")
        self.verticalLayout_3.addWidget(self.tabWidget)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setMinimumSize(QtCore.QSize(300, 120))
        self.scrollArea.setMaximumSize(QtCore.QSize(300, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 298, 118))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_info = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_info.setMinimumSize(QtCore.QSize(0, 0))
        self.label_info.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_info.setOpenExternalLinks(False)
        self.label_info.setObjectName("label_info")
        self.verticalLayout_4.addWidget(self.label_info)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout_3.addWidget(self.scrollArea)
        self.pushButton_run = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_run.setEnabled(False)
        self.pushButton_run.setMinimumSize(QtCore.QSize(300, 0))
        self.pushButton_run.setMaximumSize(QtCore.QSize(300, 16777215))
        self.pushButton_run.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_run.setAutoDefault(False)
        self.pushButton_run.setDefault(False)
        self.pushButton_run.setObjectName("pushButton_run")
        self.verticalLayout_3.addWidget(self.pushButton_run)
        self.pushButton_stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop.setEnabled(False)
        self.pushButton_stop.setMinimumSize(QtCore.QSize(300, 0))
        self.pushButton_stop.setMaximumSize(QtCore.QSize(300, 16777215))
        self.pushButton_stop.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_stop.setIconSize(QtCore.QSize(10, 10))
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.verticalLayout_3.addWidget(self.pushButton_stop)
        self.gridLayout_4.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider_track.valueChanged['int'].connect(self.label_reso_track.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Settings"))
        self.groupBox.setTitle(_translate("MainWindow", "Source"))
        self.label_5.setText(_translate("MainWindow", "From camera"))
        self.label_7.setText(_translate("MainWindow", "From file"))
        self.pushButton_del.setText(_translate("MainWindow", "del"))
        self.comboBox_cam.setItemText(1, _translate("MainWindow", "Camera 0"))
        self.comboBox_cam.setItemText(2, _translate("MainWindow", "Camera 1"))
        self.pushButton_file.setText(_translate("MainWindow", "Search"))
        self.label_2.setText(_translate("MainWindow", "Classes"))
        self.label.setText(_translate("MainWindow", "Model"))
        self.comboBox_model.setItemText(0, _translate("MainWindow", "COCO"))
        self.comboBox_model.setItemText(1, _translate("MainWindow", "yolov3_lata"))
        self.pushButton_info.setText(_translate("MainWindow", "i"))
        self.label_3.setText(_translate("MainWindow", "Resolution"))
        self.label_reso_det.setText(_translate("MainWindow", "416"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_set_det), _translate("MainWindow", "Detector settings"))
        self.label_4.setText(_translate("MainWindow", "Tracker"))
        self.comboBox_tracker.setItemText(0, _translate("MainWindow", "csrt"))
        self.comboBox_tracker.setItemText(1, _translate("MainWindow", "kcf"))
        self.comboBox_tracker.setItemText(2, _translate("MainWindow", "boosting"))
        self.comboBox_tracker.setItemText(3, _translate("MainWindow", "mil"))
        self.comboBox_tracker.setItemText(4, _translate("MainWindow", "tld"))
        self.comboBox_tracker.setItemText(5, _translate("MainWindow", "medianflow"))
        self.comboBox_tracker.setItemText(6, _translate("MainWindow", "mosse"))
        self.label_6.setText(_translate("MainWindow", "Resolution"))
        self.label_reso_track.setText(_translate("MainWindow", "160"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_set_track), _translate("MainWindow", "Tracker settings"))
        self.label_info.setText(_translate("MainWindow", "Please, select a source"))
        self.pushButton_run.setText(_translate("MainWindow", "RUN"))
        self.pushButton_stop.setText(_translate("MainWindow", "STOP"))

