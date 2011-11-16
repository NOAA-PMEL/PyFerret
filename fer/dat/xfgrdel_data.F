        BLOCK DATA FGRDEL_DATA

* initialization of COMMON / FGRDEL /
        IMPLICIT NONE
        INCLUDE 'fgrdel.cmn'

        INTEGER totmaxcolorobjs
        PARAMETER (totmaxcolorobjs = maxwindowobjs * maxcolorobjs)
        INTEGER totmaxbrushobjs
        PARAMETER (totmaxbrushobjs = maxwindowobjs * maxbrushobjs)
        INTEGER totmaxfontobjs
        PARAMETER (totmaxfontobjs = maxwindowobjs * maxfontobjs)
        INTEGER totmaxpenobjs
        PARAMETER (totmaxpenobjs = maxwindowobjs * maxpenobjs)
        INTEGER totmaxsymbolobjs
        PARAMETER (totmaxsymbolobjs = maxwindowobjs * maxsymbolobjs)

* Window objects (bindings to a window)
        DATA windowobjs / maxwindowobjs * nullobj /

* Color objects for each window
        DATA colorobjs / totmaxcolorobjs * nullobj /

* Brush objects for each window
        DATA brushobjs / totmaxbrushobjs * nullobj /

* Font objects for each window
        DATA fontobjs / totmaxfontobjs * nullobj /

* Pen objects for each window
        DATA penobjs / totmaxpenobjs * nullobj /

* Symbol objects for each window
        DATA symbolobjs / totmaxsymbolobjs * nullobj /

* Initially no active window
        DATA activewindow / 0 /

* Initially the implicit identity transform is active
        DATA activetransform / 0 /

* Initially no active brush, pen, symbol
        DATA activebrush / 0 /
        DATA activepen / 0 /
        DATA activesymbol / 0 /

* The brush used if no brush is active - initialize to solid black
        DATA lastbrushcolor / 2 /
        DATA lastbrushstyle / 'solid' /
        DATA lastbstylen / 5 /

* Nothing has been drawn to the current view of the active window
        DATA somethingdrawn / .FALSE. /

* Display the window on creation and updates.
* Window ID 9 is for creating animations, thus not shown by default.
        DATA showwindow / .TRUE., .TRUE., .TRUE.,
     .                    .TRUE., .TRUE., .TRUE.,
     .                    .TRUE., .TRUE., .FALSE. /

* Default factors for normalized device coordinates.
        DATA ndcxmax / maxwindowobjs * 1.0 /
        DATA ndcymax / maxwindowobjs * 1.0 /

* Coordinate transformation data independent of windows
        DATA viewleftfrac / maxtransforms * 0.0 /
        DATA viewbottomfrac / maxtransforms * 0.0 /
        DATA viewrightfrac / maxtransforms * 1.0 /
        DATA viewtopfrac / maxtransforms * 1.0 /
        DATA viewleftcoord / maxtransforms * 0.0 /
        DATA viewbottomcoord / maxtransforms * 0.0 /
        DATA viewrightcoord / maxtransforms * 1.0 /
        DATA viewtopcoord / maxtransforms * 1.0 /

* Default window sizes.
        DATA windowwidth / maxwindowobjs * 7.5 /
        DATA windowheight / maxwindowobjs * 5.625 /

* Default color object number to use to clear the window;
* 1 = white, 2 = black
        DATA clearcolor / 1, 1, 1, 1, 1, 1, 1, 1, 2 /

* Values when creating windows.
* These default values (especially windowtitle) may be reset.
        DATA enginename / 'PyQtPipedViewer' /
        DATA windowtitle / 'FERRET' /

        END

