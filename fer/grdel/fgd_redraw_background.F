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

        INTEGER colornum, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_REDRAW_BACKGROUND: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_REDRAW_BACKGROUND: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Background color is always color zero = colornum one
        colornum = 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_REDRAW_BACKGROUND: invalid colorindex'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Redraw the contents of the window with this background color
        CALL FGDWINREDRAW(success, windowobjs(windowid), 
     .                    colorobjs(colornum, windowid))
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

        RETURN
        END

