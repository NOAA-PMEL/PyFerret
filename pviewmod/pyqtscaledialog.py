'''
Dialog for obtaining scaling information from the user.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

from PyQt4.QtCore import SIGNAL
from PyQt4.QtGui  import QApplication, QDialog, QDialogButtonBox, \
                         QGridLayout, QLabel, QLineEdit, QMessageBox


class PyQtScaleDialog(QDialog):
    '''
    Dialog for obtaining scaling information from the user.
    Validates that the resulting width and height values
    are larger than specified minimums.
    '''

    FLTSTR_FORMAT = "%#6.2f"

    def __init__(self, title, message, scale, width, height,
                 minwidth, minheight, parent=None):
        '''
        Creates a scaling dialog with title as the window title,
        message as the dialog message, and scale as the current
        scaling value which gives a pixmap of size width and
        height.  The minimum acceptable width and heights are
        given by minwidth and minheight.
        '''
        super(PyQtScaleDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.__scale = scale
        self.__width = width
        self.__height = height
        self.__minwidth = minwidth
        self.__minheight = minheight

        messagelabel = QLabel(message, self)
        scalelabel = QLabel(self.tr("&Scale"), self)
        self.__scaleedit = QLineEdit(self.FLTSTR_FORMAT % scale, self)
        scalelabel.setBuddy(self.__scaleedit)
        widthlabel = QLabel(self.tr("Width   (min %1)  ") \
                                 .arg(self.FLTSTR_FORMAT % minwidth), self)
        self.__widthlabel = QLabel(self.FLTSTR_FORMAT % width, self)
        heightlabel = QLabel(self.tr("Height  (min %1)  ") \
                                 .arg(self.FLTSTR_FORMAT % minheight), self)
        self.__heightlabel = QLabel(self.FLTSTR_FORMAT % height, self)
        buttonbox = QDialogButtonBox(QDialogButtonBox.Ok |
                                     QDialogButtonBox.Cancel |
                                     QDialogButtonBox.Reset)

        layout = QGridLayout()
        layout.addWidget(messagelabel, 0, 0, 1, 2)
        layout.addWidget(scalelabel, 1, 0)
        layout.addWidget(self.__scaleedit, 1, 1)
        layout.addWidget(widthlabel, 2, 0)
        layout.addWidget(self.__widthlabel, 2, 1)
        layout.addWidget(heightlabel, 3, 0)
        layout.addWidget(self.__heightlabel, 3, 1)
        layout.addWidget(buttonbox, 4, 0, 1, 2)
        self.setLayout(layout)

        self.connect(self.__scaleedit, SIGNAL("textChanged(QString)"), self.updateValues)
        self.connect(buttonbox, SIGNAL("accepted()"), self.checkValues)
        self.connect(buttonbox, SIGNAL("rejected()"), self.reject)
        resetbutton = buttonbox.button(QDialogButtonBox.Reset)
        self.connect(resetbutton, SIGNAL("clicked()"), self.resetValues)

    def updateValues(self, newstring):
        (newscale, okay) = newstring.toFloat()
        if okay:
            newwidth = self.__width * newscale / self.__scale
            self.__widthlabel.setText(self.FLTSTR_FORMAT % newwidth)
            newheight = self.__height * newscale / self.__scale
            self.__heightlabel.setText(self.FLTSTR_FORMAT % newheight)

    def checkValues(self):
        okay = self.getValues()[1]
        if okay:
            self.accept()
        else:
            QMessageBox.warning(self, self.tr("Invalid value"),
                                self.tr("Scale value is not valid"))

    def getValues(self):
        (newscale, okay) = self.__scaleedit.text().toFloat()
        newwidth = self.__width * newscale / self.__scale
        newheight = self.__height * newscale / self.__scale
        if (not okay) or (newwidth < self.__minwidth) \
                      or (newheight < self.__minheight):
            return (0.0, False)
        return (newscale, True)

    def resetValues(self):
        self.__scaleedit.setText(self.FLTSTR_FORMAT % self.__scale)
        self.__widthlabel.setText(self.FLTSTR_FORMAT % self.__width)
        self.__heightlabel.setText(self.FLTSTR_FORMAT % self.__height)


if __name__ == "__main__":
    app = QApplication(["tester"])
    resizedialog = PyQtScaleDialog("Scale Dialog",
                                    "Message of the scale dialog",
                                    1.0, 15.035, 32.044, 0.5, 0.5)
    retval = resizedialog.exec_()
    print "retval = %d" % retval
    if retval == QDialog.Accepted:
        rettuple = resizedialog.getValues()
        print "getValues returned: %s" % str(rettuple)
