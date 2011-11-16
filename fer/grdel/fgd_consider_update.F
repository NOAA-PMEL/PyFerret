*
* Consider updating the active Window.  If something has
* been drawn to the active window since the last update,
* and if the program is at the base command level (not in
* a script), then issue an FGD_GUWK command on the active
* window.
*
* If the argument atapause is .TRUE., update irregardless
* of the command level (ie, waiting at a pause in a script).
*

        SUBROUTINE FGD_CONSIDER_UPDATE(atapause)

        IMPLICIT NONE
        LOGICAL atapause

        INCLUDE 'ferret.parm'
        INCLUDE 'xcontrol.cmn'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER showit, success, errmsglen
        CHARACTER*2048 errmsg

*       Return if nothing has been drawn to the active window
*       since the last update (fastest check first)
        IF ( .NOT. somethingdrawn ) THEN
            RETURN
        ENDIF
*       Return if not at a pause and not at the base command level
        IF ( (.NOT. atapause) .AND. (csp .GT. 0) ) THEN
            RETURN
        ENDIF
*       Return if there is not a valid active window
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            RETURN
        ENDIF
*       End the active view so the drawn data will show up,
*       but then start a new view in the same position with
*       the same coordinates.  Do this by reselecting the
*       current transformation.
        CALL FGD_GSELNT(activetransform)
*       Make the call to show or hide the window
        IF ( showwindow(activewindow) ) THEN
            showit = 1
        ELSE
            showit = 0
        ENDIF
        CALL FGDWINSETVIS(success, windowobjs(activewindow), showit)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

