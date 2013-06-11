*
* Create a temporary pen and symbol object (dot) 
* for the default window using the given color
*

        SUBROUTINE FGD_CREATE_TEMP_PEN(ipen, icolor, thickness)

        IMPLICIT NONE

        INTEGER ipen, icolor
        REAL*4 thickness
        REAL*4 dotsize

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: Invalid activewindow value'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: null activewindow'
        ENDIF
        IF ( (icolor .LT. 0) .OR.
     .       (icolor .GE. maxcolorobjs) ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: Invalid icolor'
        ENDIF
        IF ( colorobjs(icolor+1, activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_CREATE_TEMP_PEN: null colorobj for icolor'
        ENDIF

        ipen = maxpenobjs
*       linestyle = 1 is a solid line
        CALL FGD_GSPLR(activewindow, ipen, 1, thickness, icolor)
*       also make a temporary symbol object (dot) at the same array index
*       code in ferret expects this
        dotsize = 1.0
        CALL FGD_GSPMR(activewindow, ipen, 1, dotsize, icolor)

        RETURN
        END

