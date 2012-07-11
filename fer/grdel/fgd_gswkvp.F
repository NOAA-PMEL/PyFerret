*
* Set the size of the viewer window.
* (set workstation viewport)
* Coordinates are given in meters.
* Only the height and width are used;
* the absolute positions are ignored.
*

        SUBROUTINE FGD_GSWKVP(windowid, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER windowid
        REAL xmin, xmax, ymin, ymax

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL newwidth, newheight
        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSWKVP: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSWKVP: null windowobj'
        ENDIF

*       Compute the window width and height in inches
*       Assume a half inch minimum for the width and height
        newwidth = (xmax - xmin) * (1000.0 / 25.4)
        IF ( newwidth .LE. 0.5 ) THEN
            STOP 'FGD_GSWKVP: invalid width'
        ENDIF
        newheight = (ymax - ymin) * (1000.0 / 25.4)
        IF ( newheight .LE. 0.5 ) THEN
            STOP 'FGD_GSWKVP: invalid height'
        ENDIF
        windowwidth(windowid) = newwidth
        windowheight(windowid) = newheight

*       Resize the window - uses units of pixels
        newwidth = newwidth * windowdpix(windowid)
        newheight = newheight * windowdpiy(windowid)
        CALL FGDWINSETSIZE(success, windowobjs(windowid),
     .                     newwidth, newheight)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

