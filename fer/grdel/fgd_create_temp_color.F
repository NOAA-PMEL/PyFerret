*
* Create a temporary color object for the default window
*

        SUBROUTINE FGD_CREATE_TEMP_COLOR(windowid, icolor, 
     .                 redfrac, greenfrac, bluefrac, opaquefrac)

        IMPLICIT NONE

        INTEGER windowid, icolor
        REAL*4 redfrac, greenfrac, bluefrac, opaquefrac

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_CREATE_TEMP_COLOR: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_CREATE_TEMP_COLOR: null windowobj'
        ENDIF
        IF ( (redfrac .LT. 0.0) .OR.  (redfrac .GT. 1.0) ) THEN
            STOP 'FGD_CREATE_TEMP_COLOR: Invalid redfrac'
        ENDIF
        IF ( (greenfrac .LT. 0.0) .OR.  (greenfrac .GT. 1.0) ) THEN
            STOP 'FGD_CREATE_TEMP_COLOR: Invalid greenfrac'
        ENDIF
        IF ( (bluefrac .LT. 0.0) .OR.  (bluefrac .GT. 1.0) ) THEN
            STOP 'FGD_CREATE_TEMP_COLOR: Invalid bluefrac'
        ENDIF
        IF ( (opaquefrac .LT. 0.0) .OR.  (opaquefrac .GT. 1.0) ) THEN
            STOP 'FGD_CREATE_TEMP_COLOR: Invalid opaquefrac'
        ENDIF

        icolor = maxcolorobjs - 1
        CALL FGD_GSCR(windowid, icolor,
     .                redfrac, greenfrac, bluefrac, opaquefrac)

        RETURN
        END

