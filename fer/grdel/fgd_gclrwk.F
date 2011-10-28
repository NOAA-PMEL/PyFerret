*
* Clear the window.
* Input arguments:
*    windowid: ID number of the window to clear
*    ctrlflag: ignored
* Also used from fgrdel.cmn:
*    clearcolor(windowid): number of the color object to
*                          use to  clearing the window
*

        SUBROUTINE FGD_GCLRWK(windowid, ctrlflag)

        IMPLICIT NONE
        INTEGER windowid, ctrlflag

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GCLRWK: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GCLRWK: null windowobj'
        ENDIF
        colornum = clearcolor(windowid)
        IF ( (colornum .LT. 1) .OR.
     .       (colornum .GT. maxcolorobjs) ) THEN
            STOP 'FGD_GCLRWK: Invalid clearcolor'
        ENDIF
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GCLRWK: null clearcolor'
        ENDIF

        CALL FGDWINCLEAR(success, windowobjs(windowid),
     .                   colorobjs(colornum, windowid))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

