*
* Set the color object number to use for clearing a window
*

        SUBROUTINE FGD_BKGCOLOR(windowid, colorindex)

        IMPLICIT NONE
        INTEGER windowid, colorindex

        INCLUDE 'fgrdel.cmn'

        INTEGER colornum

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGDBKGCOLOR: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGDBKGCOLOR: null windowobj'
        ENDIF

        colornum = colorindex + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGDBKGCOLOR: invalid colorindex'
        ENDIF

        clearcolor(windowid) = colornum

        RETURN
        END

