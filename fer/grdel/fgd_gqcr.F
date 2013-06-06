*
* Returns the RGBA fractions of a color object with index
* colorindex for the window with ID number windowid.  
* If the color object exists the RGBA fractions are returned 
* in redfrac, greenfrac, bluefrac, and opaquefrac arguments, 
* and istat is set to zero.  If the colorobj does not exist, 
* then istat is set to one and the RGBA fractions are unchanged.
*

        SUBROUTINE FGD_GQCR(windowid, colorindex, istat,
     .                      redfrac, greenfrac, bluefrac, opaquefrac)

        IMPLICIT NONE
        INTEGER windowid, colorindex, istat
        REAL*4 redfrac, greenfrac, bluefrac, opaquefrac

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GQCR: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GQCR: null windowobj'
        ENDIF
        IF ( (colorindex .LT. 0) .OR.
     .       (colorindex .GE. maxcolorobjs) ) THEN
            STOP 'FGD_GQCR: Invalid colorindex'
        ENDIF

        colornum = colorindex + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            istat = 1
            RETURN
        ENDIF

*       Simplest approach for retrieving color RGBA values was 
*       just to store them in the common block when they are set
        redfrac = colorrgba(1, colornum, windowid)
        greenfrac = colorrgba(2, colornum, windowid)
        bluefrac = colorrgba(3, colornum, windowid)
        opaquefrac = colorrgba(4, colornum, windowid)
        istat = 0

        RETURN
        END

