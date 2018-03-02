*
* Clear the window.
* Input arguments:
*    windowid: ID number of the window to clear
*    ctrlflag: ignored
*
* NOTE: if windowid is not valid, this call is ignored
*       since this can arise from SET WINDOW /NEW /CLEAR
*

        SUBROUTINE FGD_GCLRWK(windowid, ctrlflag)

        IMPLICIT NONE
        INTEGER windowid, ctrlflag

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            RETURN
        ENDIF
*       Background color is always color zero which is colornum one
        colornum = 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GCLRWK: invalid color for background'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        CALL FGDWINCLEAR(success, windowobjs(windowid),
     .                   colorobjs(colornum, windowid))
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

        RETURN
        END

