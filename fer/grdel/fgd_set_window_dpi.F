*
*

        SUBROUTINE FGD_SET_WINDOW_DPI(success, windowid, newdpi)

        IMPLICIT NONE

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, windowid
        REAL*4 newdpi

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_WINDOW_DPI: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_SET_WINDOW_DPI: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            success = 0
            RETURN
        ENDIF

        CALL FGDWINSETDPI(success, windowobjs(windowid), newdpi)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

        RETURN
        END
