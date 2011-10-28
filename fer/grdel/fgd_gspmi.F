*
* Set the active symbol of the active window
* (set polymarker index)
*

        SUBROUTINE FGD_GSPMI(symbolnum)

        IMPLICIT NONE
        INTEGER symbolnum

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSPMI: Invalid activewindow'
        ENDIF
        IF ( (symbolnum .LT. 1) .OR.
     .       (symbolnum .GT. maxsymbolobjs) ) THEN
            STOP 'FGD_GSPMI: Invalid symbolnum'
        ENDIF
        IF ( symbolobjs(symbolnum, activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GSPMI: null symbolobj'
        ENDIF

        activesymbol = symbolnum

        RETURN
        END

