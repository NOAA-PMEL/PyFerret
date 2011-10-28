'''
Dialog for obtaining resize information from the user.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

from PyQt4.QtCore import SIGNAL
from PyQt4.QtGui  import QApplication, QDialog, QDialogButtonBox, \
                         QGridLayout, QLabel, QLineEdit, QMessageBox

class PyQtResizeDialog(QDialog):
    '''
    Dialog for obtaining resize information from the user.
    Validates that the specified width and height values
    are larger than specified minimums.
    '''

    def __init__(self, title, message, width, height,
                 minwidth, minheight, parent=None):
        '''
        Creates a resize dialog with title as the window title,
        message as the dialog message, and width and height as
        the initial values of the respective QLineEdit widgets.  
        '''
        super(PyQtResizeDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.__width = width
        self.__height = height
        self.__minwidth = minwidth
        self.__minheight = minheight

        messagelabel = QLabel(message, self)

        widthlabel = QLabel( self.tr("&Width   (min %1)  ") \
                                 .arg("%#6.2f" % minwidth), self)
        self.__widthedit = QLineEdit(str(width), self)
        widthlabel.setBuddy(self.__widthedit)

        heightlabel = QLabel(self.tr("&Height  (min %1)  ") \
                                 .arg("%#6.2f" % minheight), self)
        self.__heightedit = QLineEdit(str(height), self)
        heightlabel.setBuddy(self.__heightedit)

        buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | 
                                     QDialogButtonBox.Cancel |
                                     QDialogButtonBox.Reset)
        
        layout = QGridLayout()
        layout.addWidget(messagelabel, 0, 0, 1, 2)
        layout.addWidget(widthlabel, 1, 0)
        layout.addWidget(self.__widthedit, 1, 1)
        layout.addWidget(heightlabel, 2, 0)
        layout.addWidget(self.__heightedit, 2, 1)
        layout.addWidget(buttonbox, 3, 0, 1, 2)
        self.setLayout(layout)

        self.connect(buttonbox, SIGNAL("accepted()"), self.checkValues)
        self.connect(buttonbox, SIGNAL("rejected()"), self.reject)
        resetbutton = buttonbox.button(QDialogButtonBox.Reset)
        self.connect(resetbutton, SIGNAL("clicked()"), self.resetValues)

    def checkValues(self):
        okay = self.getValues()[2]
        if okay:
            self.accept()
        else:
            QMessageBox.warning(self, self.tr("Invalid value"), 
                                self.tr("Values are not valid"))

    def getValues(self):
        (newwidth, okay) = self.__widthedit.text().toFloat()
        if (not okay) or (newwidth < self.__minwidth):
            return (0.0, 0.0, False)
        (newheight, okay) = self.__heightedit.text().toFloat()
        if (not okay) or (newheight <= self.__minheight):
            return (0.0, 0.0, False)
        return (newwidth, newheight, True)

    def resetValues(self):
        self.__widthedit.setText(str(self.__width))
        self.__heightedit.setText(str(self.__height))


if __name__ == "__main__":
    width = 15.05
    height = 32.45
    app = QApplication(["tester"])
    resizedialog = PyQtResizeDialog("Resize Dialog",
                                    "Message of the resize dialog",
                                    width, height, 0.5, 10)
    retval = resizedialog.exec_()
    print "retval = %d" % retval
    if retval == QDialog.Accepted:
        rettuple = resizedialog.getValues()
        print "getValues returned: %s" % str(rettuple)
