*
* Set the active window
*

        SUBROUTINE FGD_GACWK(windowid)

        IMPLICIT NONE
        INTEGER windowid
        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GACWK: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GACWK: null windowobj'
        ENDIF

        activewindow = windowid

*       Begin a view in the active window by selecting
*       the default transform (same as windowid)
        CALL FGD_GSELNT(windowid)

        RETURN
        END

