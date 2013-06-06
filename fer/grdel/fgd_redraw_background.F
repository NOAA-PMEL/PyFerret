*
* Redraw the current image using the current color zero
* (colornum one) as the background color.
*

        SUBROUTINE FGD_REDRAW_BACKGROUND(windowid)

        IMPLICIT NONE
        INTEGER windowid, colorindex

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_REDRAW_BACKGROUND: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_REDRAW_BACKGROUND: null windowobj'
        ENDIF

*       Background color is always color zero = colornum one
        colornum = 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGD_REDRAW_BACKGROUND: invalid colorindex'
        ENDIF

*       Redraw the contents of the window with this background color
        CALL FGDWINREDRAW(success, windowobjs(windowid), 
     .                    colorobjs(colornum, windowid))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

