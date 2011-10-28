*
* Returns the RGB fractions of a color object with index
* colorindex for the window with ID number windowid.  
* If the color object exists the RGB fractions are returned 
* in redfrac, greenfrac, and bluefrac arguments, and istat
* is set to zero.  If the colorobj does not exist, then
* istat is set to one and the RGB fractions are unchanged.
* The argument rettype is ignored.
*

        SUBROUTINE FGD_GQCR(windowid, colorindex, rettype,
     .                  istat, redfrac, greenfrac, bluefrac)

        IMPLICIT NONE
        INTEGER windowid, colorindex, rettype, istat
        REAL redfrac, greenfrac, bluefrac

        INCLUDE 'ferret.parm'
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

        redfrac = colorrgb(1, colornum, windowid)
        greenfrac = colorrgb(2, colornum, windowid)
        bluefrac = colorrgb(3, colornum, windowid)
        istat = 0

        RETURN
        END
