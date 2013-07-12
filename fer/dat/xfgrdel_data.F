        BLOCK DATA FGRDEL_DATA

* initialization of COMMON / FGRDEL /
        IMPLICIT NONE
        INCLUDE 'plot_setup.parm' ! dflt_widht, dflt_height, dflt_thicken
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

* By default clip drawings to the current view
        DATA cliptoview / maxwindowobjs * .TRUE. /

* Default factors for normalized device coordinates.
        DATA ndcxmax / maxwindowobjs * 1.0 /
        DATA ndcymax / maxwindowobjs * 1.0 /

* "Uninitialized" DPI values
        DATA windowdpix / maxwindowobjs * 0.0 /
        DATA windowdpiy / maxwindowobjs * 0.0 /

* Scaling factor for line widths and symbol sizes
        DATA thickfactor / maxwindowobjs * dflt_thicken /

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

