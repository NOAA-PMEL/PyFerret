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

        INTEGER errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SET_WINDOW_DPI: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_SET_WINDOW_DPI: null windowobj'
        ENDIF

        CALL FGDWINSETDPI(success, windowobjs(windowid), newdpi)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END
