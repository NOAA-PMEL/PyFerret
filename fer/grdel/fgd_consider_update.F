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

        INCLUDE 'xcontrol.cmn'
        INCLUDE 'fgrdel.cmn'

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
*       Update the display
        CALL FGD_GUWK(activewindow)

        RETURN
        END

