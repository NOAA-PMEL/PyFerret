*
* Create a temporary pen and symbol object (dot) 
* for the default window using the given color
*

        SUBROUTINE FGD_CREATE_TEMP_PEN(windowid, 
     .                        ipen, icolor, thickness)

        IMPLICIT NONE

        INTEGER windowid, ipen, icolor
        REAL*4 thickness
        REAL*4 dotsize
        CHARACTER*3 dotmark

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: null windowobj'
        ENDIF
        IF ( (icolor .LT. 0) .OR.
     .       (icolor .GE. maxcolorobjs) ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: Invalid icolor'
        ENDIF
        IF ( colorobjs(icolor+1, windowid) .EQ. nullobj ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: null colorobj for icolor'
        ENDIF

        ipen = maxpenobjs
*       linestyle = 1 is a solid line
        CALL FGD_GSPLR(windowid, ipen, 1, thickness, icolor)
*       also make a temporary symbol object (dot) at the same array index;
*       code in ferret expects this
        dotmark = 'dot'
        dotsize = 6.0 + thickness
        CALL FGD_GSPMR(windowid, ipen, dotmark, dotsize, icolor)

        RETURN
        END

