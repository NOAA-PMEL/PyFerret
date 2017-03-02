'''
Dialog for obtaining scaling information from the user.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

from __future__ import print_function

import sys

if sys.version_info[0] > 2:
    # First try to import PyQt5, then try PyQt4 if that fails
    try:
        import PyQt5
        QT_VERSION = 5
    except ImportError:
        import PyQt4
        QT_VERSION = 4
else:
    # PyQt5 requires Python3.x, so only try PyQt4
    import PyQt4
    QT_VERSION = 4

# Now that the PyQt version is determined, import the parts
# allowing any import errors to propagate out
if QT_VERSION == 5:
    from PyQt5.QtCore    import Qt
    from PyQt5.QtWidgets import QApplication, QButtonGroup, QDialog, \
                                QDialogButtonBox, QGridLayout, QGroupBox, \
                                QLabel, QLineEdit, QMessageBox, QRadioButton
else:
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui  import QApplication, QButtonGroup, QDialog, \
                             QDialogButtonBox, QGridLayout, QGroupBox, \
                             QLabel, QLineEdit, QMessageBox, QRadioButton


class ScaleDialogPQ(QDialog):
    '''
    Dialog for obtaining scaling information from the user.
    Validates that the resulting width and height values
    are not smaller than the specified minimums.
    '''

    def __init__(self, scale, width, height,
                 minwidth, minheight, autoscale, parent=None):
        '''
        Creates a scaling dialog, with scale as the current
        scaling value which gives a pixmap of size width and
        height.  The minimum acceptable width and heights are
        given by minwidth and minheight.  Values are assumed to
        be in units of pixels.  The value of autoscale sets the
        default value of "Scale image to fir window frame".
        '''
        super(ScaleDialogPQ, self).__init__(parent)

        self.__scale = float(scale)
        self.__pixwidth = float(width)
        self.__inchwidth = float(width) / float(self.physicalDpiX())
        self.__pixheight = float(height)
        self.__inchheight = float(height) / float(self.physicalDpiY())
        self.__minpixwidth = int(minwidth)
        self.__minpixheight = int(minheight)
        self.__autoscale = bool(autoscale)

        self.FLTSTR_FORMAT = "%#.3f"

        self.setWindowTitle(self.tr("Image Size Scaling"))

        # auto-scaling option at the top
        autoscalelabel = QLabel(self.tr("Scale image to fit window frame?"), 
                                self)
        autoscalelabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.__autoyesbtn = QRadioButton(self.tr("&Yes"), self)
        self.__autonobtn = QRadioButton(self.tr("&No"), self)
        autoscalebtngrp = QButtonGroup(self)
        autoscalebtngrp.addButton(self.__autoyesbtn)
        autoscalebtngrp.addButton(self.__autonobtn)

        # put the manual scaling settings into their own box
        self.__grpbox = QGroupBox(self.tr("Fixed scaling"), self)

        # create the widgets going inside this group box
        messagelabel = QLabel(
            self.tr("Scaling factor (both horiz. and vert.) for the image"), 
            self.__grpbox)
        messagelabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        scalelabel = QLabel(self.tr("&Scale: "), self.__grpbox)
        self.__scaleedit = QLineEdit(self.FLTSTR_FORMAT % self.__scale, 
                                     self.__grpbox)
        scalelabel.setBuddy(self.__scaleedit)

        widthbegin = QLabel(self.tr("Width: "), self.__grpbox)
        self.__pixwidthlabel = QLabel(str(int(self.__pixwidth + 0.5)), 
                                      self.__grpbox)
        widthmiddle = QLabel(self.tr("pixels, or"), self.__grpbox)
        self.__inchwidthlabel = QLabel(self.FLTSTR_FORMAT % self.__inchwidth, 
                                       self.__grpbox)
        widthend = QLabel(self.tr("inches on the screen"), self.__grpbox)
        minwidthlabel = QLabel(self.tr("(must not be less than %d pixels)" % \
                               self.__minpixwidth), self.__grpbox)

        heightbegin = QLabel(self.tr("Height:"), self.__grpbox)
        self.__pixheightlabel = QLabel(str(int(self.__pixheight + 0.5)), 
                                       self.__grpbox)
        heightmiddle = QLabel(self.tr("pixels, or"), self.__grpbox)
        self.__inchheightlabel = QLabel(self.FLTSTR_FORMAT % self.__inchheight, 
                                        self.__grpbox)
        heightend = QLabel(self.tr("inches on the screen"), self.__grpbox)
        minheightlabel = QLabel(self.tr("(must not be less than %d pixels)" % \
                                self.__minpixheight), self.__grpbox)

        # layout the widgets in this group box
        layout = QGridLayout()

        layout.addWidget(messagelabel, 0, 0, 1, 5)

        layout.addWidget(scalelabel, 1, 0, 1, 1)
        layout.addWidget(self.__scaleedit, 1, 1, 1, 4)

        layout.addWidget(widthbegin, 2, 0, 1, 1)
        layout.addWidget(self.__pixwidthlabel, 2, 1, 1, 1)
        layout.addWidget(widthmiddle, 2, 2, 1, 1)
        layout.addWidget(self.__inchwidthlabel, 2, 3, 1, 1)
        layout.addWidget(widthend, 2, 4, 1, 1)

        layout.addWidget(minwidthlabel, 3, 1, 1, 4)

        layout.addWidget(heightbegin, 4, 0, 1, 1)
        layout.addWidget(self.__pixheightlabel, 4, 1, 1, 1)
        layout.addWidget(heightmiddle, 4, 2, 1, 1)
        layout.addWidget(self.__inchheightlabel, 4, 3, 1, 1)
        layout.addWidget(heightend, 4, 4, 1, 1)

        layout.addWidget(minheightlabel, 5, 1, 1, 4)

        # assign this layout to the group box
        self.__grpbox.setLayout(layout)

        # layout the widgets in the dialog (outside the group box)
        layout = QGridLayout()
        layout.addWidget(autoscalelabel, 0, 0, 1, 1)
        layout.addWidget(self.__autoyesbtn, 0, 1, 1, 1)
        layout.addWidget(self.__autonobtn, 0, 2, 1, 1)
        layout.addWidget(self.__grpbox, 1, 0, 1, 3)
        
        buttonbox = QDialogButtonBox(QDialogButtonBox.Ok |
                                     QDialogButtonBox.Cancel |
                                     QDialogButtonBox.Reset,
                                     Qt.Horizontal, self)

        layout.addWidget(buttonbox, 2, 0, 1, 3)

        self.setLayout(layout)

        # The OK button is not the default here in Qt4.2
        okbutton = buttonbox.button(QDialogButtonBox.Ok)
        okbutton.setDefault(True)

        resetbutton = buttonbox.button(QDialogButtonBox.Reset)

        self.__autoyesclicked = self.__autoyesbtn.clicked
        self.__autoyesclicked.connect(self.setAutoScale)

        self.__autonoclicked = self.__autonobtn.clicked
        self.__autonoclicked.connect(self.unsetAutoScale)

        self.__scaletextchanged = self.__scaleedit.textChanged
        self.__scaletextchanged.connect(self.updateValues)

        self.__buttonboxaccepted = buttonbox.accepted
        self.__buttonboxaccepted.connect(self.checkValues)

        self.__buttonboxrejected = buttonbox.rejected
        self.__buttonboxrejected.connect(self.reject)

        self.__resetbuttonclicked = resetbutton.clicked
        self.__resetbuttonclicked.connect(self.resetValues)

        # initialize the state from autoscale
        if self.__autoscale:
            self.__autoyesbtn.setChecked(True)
            self.setAutoScale(True)
        else:
            self.__autonobtn.setChecked(True)
            self.unsetAutoScale(True)

    def setAutoScale(self, checked):
        if checked:
            self.__grpbox.setEnabled(False)

    def unsetAutoScale(self, checked):
        if checked:
            self.__grpbox.setEnabled(True)
            self.__scaleedit.setFocus()
            self.__scaleedit.selectAll()

    def updateValues(self, newstring):
        try:
            newscale = float(newstring)
            if (newscale < 0.0001) or (newscale > 10000.0):
                raise OverflowError()
            newval = self.__pixwidth * newscale / self.__scale
            self.__pixwidthlabel.setText(str(int(newval + 0.5)))
            newval = self.__inchwidth * newscale / self.__scale
            self.__inchwidthlabel.setText(self.FLTSTR_FORMAT % newval)
            newval = self.__pixheight * newscale / self.__scale
            self.__pixheightlabel.setText(str(int(newval + 0.5)))
            newval = self.__inchheight * newscale / self.__scale
            self.__inchheightlabel.setText(self.FLTSTR_FORMAT % newval)
        except Exception:
            pass

    def checkValues(self):
        okay = self.getValues()[2]
        if okay:
            self.accept()
        else:
            QMessageBox.warning(self, self.tr("Invalid value"),
                                self.tr("Scale value is not valid"))

    def getValues(self):
        if self.__autoyesbtn.isChecked():
            return (0.0, True, True)
        try:
            newscale = float(self.__scaleedit.text())
            if (newscale < 0.0001) or (newscale > 10000.0):
                raise OverflowError()
            newwidth = self.__pixwidth * newscale / self.__scale
            newwidth = int(newwidth + 0.5)
            newheight = self.__pixheight * newscale / self.__scale
            newheight = int(newheight + 0.5)
            if (newwidth < self.__minpixwidth) or (newheight < self.__minpixheight):
                raise OverflowError()
        except Exception:
            return (0.0, False, False)
        return (newscale, False, True)

    def resetValues(self):
        self.__scaleedit.setText(self.FLTSTR_FORMAT % self.__scale)
        self.__pixwidthlabel.setText(str(int(self.__pixwidth + 0.5)))
        self.__inchwidthlabel.setText(self.FLTSTR_FORMAT % self.__inchwidth)
        self.__pixheightlabel.setText(str(int(self.__pixheight + 0.5)))
        self.__inchheightlabel.setText(self.FLTSTR_FORMAT % self.__inchheight)
        if self.__autoscale:
            self.__autoyesbtn.setChecked(True)
            self.setAutoScale(True)
        else:
            self.__autonobtn.setChecked(True)
            self.unsetAutoScale(True)


def _test_scaledialogpq():
    app = QApplication(["tester"])
    resizedialog = ScaleDialogPQ(1.0, 500, 300, 75, 50, False)
    retval = resizedialog.exec_()
    print("retval = %d" % retval)
    if retval == QDialog.Accepted:
        rettuple = resizedialog.getValues()
        print("getValues returned: %s" % str(rettuple))

if __name__ == "__main__":
    _test_scaledialogpq()

