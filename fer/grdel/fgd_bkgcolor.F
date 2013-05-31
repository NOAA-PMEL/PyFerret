*
* Set the color object number to use for clearing a window
* Redraw the current image using this color as the background color.
*

        SUBROUTINE FGD_BKGCOLOR(windowid, colorindex)

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
            STOP 'FGDBKGCOLOR: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGDBKGCOLOR: null windowobj'
        ENDIF

        colornum = colorindex + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGDBKGCOLOR: invalid colorindex'
        ENDIF

*       Save the background/clearing color for this window
        clearcolor(windowid) = colornum

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

