*
* Return the number of dots per inch in the horizontal (X)
* and vertical (Y) directions.
*

        SUBROUTINE FGD_GETDPI(windowid, dpix, dpiy)

        IMPLICIT NONE
        INTEGER windowid
        REAL dpix, dpiy

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GETDPI: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GETDPI: null windowobj'
        ENDIF
        IF ( (windowdpix(windowid) .EQ. 0.0) .OR.
     .       (windowdpiy(windowid) .EQ. 0.0) ) THEN
            STOP 'FGD_GETDPI: Uninitialized window DPI'
        ENDIF

        dpix = windowdpix(windowid)
        dpiy = windowdpiy(windowid)

        RETURN
        END

