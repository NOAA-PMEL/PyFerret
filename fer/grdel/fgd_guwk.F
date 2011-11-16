*
* "Update" a Window.  If windowid is the active window, any
* active view is ended and restarted.  Calls FGDWINSETVIS to
* make sure the window is or is not visible.  Actual update
* of the graphics window is left to the normal operation of
* the graphics engine.
*
* Input arguments:
*    windowid: window ID number to update
*    regenflag: ignored
*

        SUBROUTINE FGD_GUWK(windowid, regenflag)

        IMPLICIT NONE
        INTEGER windowid
        INTEGER regenflag

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER showit, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GUWK: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GUWK: null windowobj'
        ENDIF

*       If the window is not the active window, any view in
*       this window should already have been ended.  But if
*       the window is the active window, and there is drawn
*       data in a view that is still active (not ended),
*       end the active view so the drawn data will show up,
*       but start a new view in the same position with the
*       same coordinates.  Do this by reselecting the current
*       transformation.
        IF ( (windowid .EQ. activewindow) .AND. somethingdrawn ) THEN
            CALL FGD_GSELNT(activetransform)
        ENDIF

*       Make the call to show or hide the window
        IF ( showwindow(windowid) ) THEN
            showit = 1
        ELSE
            showit = 0
        ENDIF
        CALL FGDWINSETVIS(success, windowobjs(windowid), showit)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

