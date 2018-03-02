*
* Set the active brush.
* (set fill area index)
*

        SUBROUTINE FGD_GSFAI(brushnum)

        IMPLICIT NONE
        INTEGER brushnum

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSFAI: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (brushnum .LT. 1) .OR.
     .       (brushnum .GT. maxbrushobjs) ) THEN
            errstr = 'FGD_GSFAI: invalid brushnum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( brushobjs(brushnum, activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSFAI: null brushobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        activebrush = brushnum

        RETURN
        END

