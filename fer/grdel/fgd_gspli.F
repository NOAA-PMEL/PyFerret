*
* Set the active pen of the active window
* (set polyline index)
*

        SUBROUTINE FGD_GSPLI(pennum)

        IMPLICIT NONE
        INTEGER pennum

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSPLI: No active window'
        ENDIF
        IF ( (pennum .LT. 1) .OR.
     .       (pennum .GT. maxpenobjs) ) THEN
            STOP 'FGD_GSPLI: Invalid pennum'
        ENDIF
        IF ( penobjs(pennum, activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GSPLI: null penobj'
        ENDIF

        activepen = pennum

        RETURN
        END

