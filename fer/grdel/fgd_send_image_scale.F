*
* Suggest to the graphics engine the initial scaling factor
* for this window. 
*

        SUBROUTINE FGD_SEND_IMAGE_SCALE(windowid, scl)

        IMPLICIT NONE

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER windowid, istat
        REAL*4 scl

        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SEND_IMAGE_SCALE: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_SEND_IMAGE_SCALE: null windowobj'
        ENDIF

        CALL FGDWINSETSCALE(success, windowobjs(windowid), scl)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END
