        BLOCK DATA FGRDEL_DATA

* initialization of COMMON / FGRDEL /
        IMPLICIT NONE
        INCLUDE 'plot_setup.parm' ! dflt_widht, dflt_height, dflt_thicken
        INCLUDE 'fgrdel.cmn'

        INTEGER totmaxcolorobjs
        PARAMETER (totmaxcolorobjs = maxwindowobjs * maxcolorobjs)
        INTEGER totmaxcolorrgba
        PARAMETER (totmaxcolorrgba = 4 * totmaxcolorobjs)
        INTEGER totmaxbrushobjs
        PARAMETER (totmaxbrushobjs = maxwindowobjs * maxbrushobjs)
        INTEGER totmaxfontobjs
        PARAMETER (totmaxfontobjs = maxwindowobjs * maxfontobjs)
        INTEGER totmaxpenobjs
        PARAMETER (totmaxpenobjs = maxwindowobjs * maxpenobjs)
        INTEGER totmaxsymbolobjs
        PARAMETER (totmaxsymbolobjs = maxwindowobjs * maxsymbolobjs)
        REAL*4 negoneflt
        PARAMETER (negoneflt = -1.0)
        REAL*4 zeroflt
        PARAMETER (zeroflt = 0.0)
        REAL*4 posoneflt
        PARAMETER (posoneflt = 1.0)

* Window objects (bindings to a window)
        DATA windowobjs / maxwindowobjs * nullobj /

* Color objects for each window
        DATA colorobjs / totmaxcolorobjs * nullobj /
        DATA colorrgba / totmaxcolorrgba * negoneflt /

* Brush objects for each window
        DATA brushobjs / totmaxbrushobjs * nullobj /
        DATA brushcolor /totmaxbrushobjs * 0 /

* Font objects for each window
        DATA fontobjs / totmaxfontobjs * nullobj /

* Pen objects for each window
        DATA penobjs / totmaxpenobjs * nullobj /
        DATA pencolor /totmaxpenobjs * 0 /
        DATA penwidth /totmaxpenobjs * negoneflt /

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

* By default clip drawings to the current view
        DATA cliptoview / maxwindowobjs * .TRUE. /

* Default factors for normalized device coordinates.
        DATA ndcxmax / maxwindowobjs * posoneflt /
        DATA ndcymax / maxwindowobjs * posoneflt /

* "Uninitialized" DPI values
        DATA windowdpix / maxwindowobjs * zeroflt /
        DATA windowdpiy / maxwindowobjs * zeroflt /

* Scaling factor for line widths and marker sizes
        DATA thickfactor / maxwindowobjs * dflt_thicken /

* Width of outlines for rectangles and other polygons
        DATA outlinewidth / maxwindowobjs * zeroflt /

* Coordinate transformation data independent of windows
        DATA viewleftfrac / maxtransforms * zeroflt /
        DATA viewbottomfrac / maxtransforms * zeroflt /
        DATA viewrightfrac / maxtransforms * posoneflt /
        DATA viewtopfrac / maxtransforms * posoneflt /
        DATA viewleftcoord / maxtransforms * zeroflt /
        DATA viewbottomcoord / maxtransforms * zeroflt /
        DATA viewrightcoord / maxtransforms * posoneflt /
        DATA viewtopcoord / maxtransforms * posoneflt /

* Default window sizes.
        DATA windowwidth / maxwindowobjs * dflt_xinches /
        DATA windowheight / maxwindowobjs * dflt_yinches /

* Default engine for a new window.
* Should not change after start-up.
* Only reset by command-line options.
        DATA defaultenginename / 'PipedViewerPQ' /

* Transparent background for save-on-exit (-batch)?
* Should not change after start-up.
* Only reset by the "-transparent" command-line option.
* Has nothing to do with "frame /transparent".
        DATA savetransparent / .FALSE. /

* Actual window values - may get reset by the user.
        DATA enginename / maxwindowobjs * 'PipedViewerPQ' /
        DATA antialias / maxwindowobjs * .TRUE. /

* Window title assigned just before creating a window
        DATA windowtitle / 'FERRET' /

        END

