*
* Set the active brush.
* (set fill area index)
*

        SUBROUTINE FGD_GSFAI(brushnum)

        IMPLICIT NONE
        INTEGER brushnum

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSFAI: No active window'
        ENDIF
        IF ( (brushnum .LT. 1) .OR.
     .       (brushnum .GT. maxbrushobjs) ) THEN
            STOP 'FGD_GSFAI: Invalid brushnum'
        ENDIF
        IF ( brushobjs(brushnum, activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GSFAI: null brushobj'
        ENDIF

        activebrush = brushnum

        RETURN
        END

