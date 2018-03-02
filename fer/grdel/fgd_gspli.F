*
* Set the active pen of the active window
* (set polyline index)
*

        SUBROUTINE FGD_GSPLI(pennum)

        IMPLICIT NONE
        INTEGER pennum

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSPLI: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (pennum .LT. 1) .OR.
     .       (pennum .GT. maxpenobjs) ) THEN
            errstr = 'FGD_GSPLI: invalid pennum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( penobjs(pennum, activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPLI: null penobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        activepen = pennum

        RETURN
        END

