'''
Dialog for obtaining scaling information from the user.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

from PyQt4.QtCore import SIGNAL, Qt
from PyQt4.QtGui  import QApplication, QDialog, QDialogButtonBox, \
                         QGridLayout, QLabel, QLineEdit, QMessageBox


class PyQtScaleDialog(QDialog):
    '''
    Dialog for obtaining scaling information from the user.
    Validates that the resulting width and height values
    are not smaller than the specified minimums.
    '''

    FLTSTR_FORMAT = "%#6.2f"

    def __init__(self, title, message, scale, width, height,
                 minwidth, minheight, parent=None):
        '''
        Creates a scaling dialog with title as the window title,
        message as the dialog message, and scale as the current
        scaling value which gives a pixmap of size width and
        height.  The minimum acceptable width and heights are
        given by minwidth and minheight.  Values are assumed to
        be in units of pixels.
        '''
        super(PyQtScaleDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.__scale = float(scale)
        self.__pixwidth = float(width)
        self.__inchwidth = float(width) / float(self.physicalDpiX())
        self.__pixheight = float(height)
        self.__inchheight = float(height) / float(self.physicalDpiY())
        self.__minpixwidth = int(minwidth)
        self.__minpixheight = int(minheight)

        messagelabel = QLabel(message, self)
        messagelabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        scalelabel = QLabel(self.tr("&Scale "), self)
        self.__scaleedit = QLineEdit(self.FLTSTR_FORMAT % self.__scale, self)
        scalelabel.setBuddy(self.__scaleedit)

        widthbegin = QLabel(self.tr("Width "), self)
        self.__pixwidthlabel = QLabel(str(int(self.__pixwidth + 0.5)), self)
        widthmiddle = QLabel(self.tr("pixels  ("))
        self.__inchwidthlabel = QLabel(self.FLTSTR_FORMAT % self.__inchwidth)
        widthend = QLabel(self.tr("inches on the screen)"))
        minwidthlabel = QLabel(self.tr("must not be less than %1 pixels") \
                               .arg(str(self.__minpixwidth)))

        heightbegin = QLabel(self.tr("Height"), self)
        self.__pixheightlabel = QLabel(str(int(self.__pixheight + 0.5)), self)
        heightmiddle = QLabel(self.tr("pixels  ("))
        self.__inchheightlabel = QLabel(self.FLTSTR_FORMAT % self.__inchheight)
        heightend = QLabel(self.tr("inches on the screen)"))
        minheightlabel = QLabel(self.tr("must not be less than %1 pixels") \
                               .arg(str(self.__minpixheight)))

        buttonbox = QDialogButtonBox(QDialogButtonBox.Ok |
                                     QDialogButtonBox.Cancel |
                                     QDialogButtonBox.Reset,
                                     Qt.Horizontal, self)

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

        layout.addWidget(buttonbox, 6, 0, 1, 5)

        self.setLayout(layout)

        self.connect(self.__scaleedit, SIGNAL("textChanged(QString)"), self.updateValues)
        self.connect(buttonbox, SIGNAL("accepted()"), self.checkValues)
        self.connect(buttonbox, SIGNAL("rejected()"), self.reject)
        resetbutton = buttonbox.button(QDialogButtonBox.Reset)
        self.connect(resetbutton, SIGNAL("clicked()"), self.resetValues)

    def updateValues(self, newstring):
        (newscale, okay) = newstring.toFloat()
        if okay:
            newval = self.__pixwidth * newscale / self.__scale
            self.__pixwidthlabel.setText(str(int(newval + 0.5)))
            newval = self.__inchwidth * newscale / self.__scale
            self.__inchwidthlabel.setText(self.FLTSTR_FORMAT % newval)
            newval = self.__pixheight * newscale / self.__scale
            self.__pixheightlabel.setText(str(int(newval + 0.5)))
            newval = self.__inchheight * newscale / self.__scale
            self.__inchheightlabel.setText(self.FLTSTR_FORMAT % newval)

    def checkValues(self):
        okay = self.getValues()[1]
        if okay:
            self.accept()
        else:
            QMessageBox.warning(self, self.tr("Invalid value"),
                                self.tr("Scale value is not valid"))

    def getValues(self):
        (newscale, okay) = self.__scaleedit.text().toFloat()
        if not okay:
            return (0.0, False)
        newwidth = self.__pixwidth * newscale / self.__scale
        newwidth = int(newwidth + 0.5)
        newheight = self.__pixheight * newscale / self.__scale
        newheight = int(newheight + 0.5)
        if (newwidth < self.__minpixwidth) or (newheight < self.__minpixheight):
            return (0.0, False)
        return (newscale, True)

    def resetValues(self):
        self.__scaleedit.setText(self.FLTSTR_FORMAT % self.__scale)
        self.__pixwidthlabel.setText(str(int(self.__pixwidth + 0.5)))
        self.__inchwidthlabel.setText(self.FLTSTR_FORMAT % self.__inchwidth)
        self.__pixheightlabel.setText(str(int(self.__pixheight + 0.5)))
        self.__inchheightlabel.setText(self.FLTSTR_FORMAT % self.__inchheight)


if __name__ == "__main__":
    app = QApplication(["tester"])
    resizedialog = PyQtScaleDialog("Scale Dialog",
                                    "Message of the scale dialog",
                                    1.0, 500, 300, 75, 50)
    retval = resizedialog.exec_()
    print "retval = %d" % retval
    if retval == QDialog.Accepted:
        rettuple = resizedialog.getValues()
        print "getValues returned: %s" % str(rettuple)
