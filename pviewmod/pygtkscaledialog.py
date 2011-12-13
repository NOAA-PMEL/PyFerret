'''
Dialog for obtaining scaling information from the user.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

import pygtk
pygtk.require('2.0')
import gtk

class PyGtkScaleDialog(gtk.Dialog):
    '''
    Dialog for obtaining scaling information from the user.
    Validates that the resulting width and height values
    are not smaller than the specified minimums.
    '''

    def __init__(self, title, message, scale, width, height,
                 minwidth, minheight, dpis, parent):
        '''
        Creates a scaling dialog with title as the window title,
        message as the dialog message, and scale as the current
        scaling value which gives a pixmap of size width and
        height.  The minimum acceptable width and heights are
        given by minwidth and minheight.  Values are assumed to
        be in units of pixels, and dpis is a two-tuple of floats
        giving the number of pixels per inch
        '''
        super(PyGtkScaleDialog, self).__init__( title, parent,
                        gtk.DIALOG_MODAL,
                        (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
                         gtk.STOCK_OK, gtk.RESPONSE_ACCEPT) )
        self.__scale = float(scale)
        self.__pixwidth = float(width)
        self.__inchwidth = float(width) / float(dpis[0])
        self.__pixheight = float(height)
        self.__inchheight = float(height) / float(dpis[1])
        self.__minpixwidth = int(minwidth)
        self.__minpixheight = int(minheight)
        self.FLTSTR_FORMAT = "%#.2f"
        # table layout
        dlgtable = gtk.Table(6, 5, False)
        dlgtable.set_row_spacings(4)
        dlgtable.set_col_spacings(4)
        # put the table in the central area of the dialog
        dlgtable.set_border_width(4)
        self.vbox.pack_start(dlgtable)
        # first table row - centered message        
        msglabel = gtk.Label(message)
        msglabel.set_alignment(0.5, 0.5)
        dlgtable.attach(msglabel, 0, 5, 0, 1)
        # second table row - scale label and text editor
        scalelabel = gtk.Label("Scale:")
        scalelabel.set_alignment(0.0, 0.5)
        dlgtable.attach(scalelabel, 0, 1, 1, 2)
        self.__scaleedit = gtk.Entry()
        self.__scaleedit.set_editable(True)
        self.__scaleedit.set_text( self.FLTSTR_FORMAT % self.__scale )
        dlgtable.attach(self.__scaleedit, 1, 5, 1, 2)
        # third table row - width labels
        widthbegin = gtk.Label("Width:")
        widthbegin.set_alignment(0.0, 0.5)
        dlgtable.attach(widthbegin, 0, 1, 2, 3)
        self.__pixwidthlabel = gtk.Label( str(int(self.__pixwidth + 0.5)) )
        dlgtable.attach(self.__pixwidthlabel, 1, 2, 2, 3)
        widthmiddle = gtk.Label("pixels, or")
        dlgtable.attach(widthmiddle, 2, 3, 2, 3)
        self.__inchwidthlabel = gtk.Label( self.FLTSTR_FORMAT % self.__inchwidth )
        dlgtable.attach(self.__inchwidthlabel, 3, 4, 2, 3)
        widthend = gtk.Label("inches on the screen")
        dlgtable.attach(widthend, 4, 5, 2, 3)
        # fourth table row - min width info label
        minwidthlabel = gtk.Label( "(must not be less than %d pixels)" % \
                                   self.__minpixwidth )
        dlgtable.attach(minwidthlabel, 1, 5, 3, 4)
        # fifth table row - height labels
        heightbegin = gtk.Label("Height:")
        heightbegin.set_alignment(0.0, 0.5)
        dlgtable.attach(heightbegin, 0, 1, 4, 5)
        self.__pixheightlabel = gtk.Label( str(int(self.__pixheight + 0.5)) )
        dlgtable.attach(self.__pixheightlabel, 1, 2, 4, 5)
        heightmiddle = gtk.Label("pixels, or")
        dlgtable.attach(heightmiddle, 2, 3, 4, 5)
        self.__inchheightlabel = gtk.Label( self.FLTSTR_FORMAT % self.__inchheight )
        dlgtable.attach(self.__inchheightlabel, 3, 4, 4, 5)
        heightend = gtk.Label("inches on the screen")
        dlgtable.attach(heightend, 4, 5, 4, 5)
        # sixth table row - min height info label
        minheightlabel = gtk.Label( "(must not be less than %d pixels)" % \
                                   self.__minpixheight )
        dlgtable.attach(minheightlabel, 1, 5, 5, 6)
        # "show" the table and everything in it
        dlgtable.show_all()
        # set the default button
        okbutton = self.get_widget_for_response(gtk.RESPONSE_ACCEPT)
        okbutton.set_flags(gtk.CAN_DEFAULT)
        self.set_default(okbutton)
        # select all the text in the text edit
        self.__scaleedit.select_region(0, -1)
        # update all the label values when the scaling value changes
        self.__scaleedit.connect("changed", self.updateValues, None)       

    def updateValues(self, widget, data):
        '''
        Updates the label values according to the scaling value
        currently contained in the text editor. 
        '''
        newval = self.__scaleedit.get_text()
        try:
            newscale = float(newval)
        except ValueError:
            newscale = 0.0
        if newscale > 0.0:
            newval = self.__pixwidth * newscale / self.__scale
            self.__pixwidthlabel.set_text(str(int(newval + 0.5)))
            newval = self.__inchwidth * newscale / self.__scale
            self.__inchwidthlabel.set_text(self.FLTSTR_FORMAT % newval)
            newval = self.__pixheight * newscale / self.__scale
            self.__pixheightlabel.set_text(str(int(newval + 0.5)))
            newval = self.__inchheight * newscale / self.__scale
            self.__inchheightlabel.set_text(self.FLTSTR_FORMAT % newval)

    def getValues(self):
        '''
        Returns (scalefactor, okay) where:
            okay: True if the scaling factor is acceptable
            scalefactor: the new scaling factor as a float
                    if okay is True; otherwise, zero
        '''
        newval = self.__scaleedit.get_text()
        try:
            newscale = float(newval)
        except ValueError:
            newscale = 0.0
        if newscale <= 0.0:
            return (0.0, False)
        newwidth = self.__pixwidth * newscale / self.__scale
        newwidth = int(newwidth + 0.5)
        newheight = self.__pixheight * newscale / self.__scale
        newheight = int(newheight + 0.5)
        if (newwidth < self.__minpixwidth) or (newheight < self.__minpixheight):
            return (0.0, False)
        return (newscale, True)


if __name__ == "__main__":
    scaledialog = PyGtkScaleDialog("Scale Dialog",
                                   "Message of the scale dialog",
                                   1.0, 500, 300, 75, 50,
                                   (120, 120), None)
    retval = scaledialog.run()
    scaledialog.hide()
    print "retval = %d" % retval
    if retval == gtk.RESPONSE_ACCEPT:
        rettuple = scaledialog.getValues()
        print "getValues returned: %s" % str(rettuple)
    scaledialog.destroy()
