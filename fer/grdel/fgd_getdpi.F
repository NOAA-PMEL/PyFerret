*
* Return the number of dots per inch in the horizontal (X)
* and vertical (Y) directions.
*

        SUBROUTINE FGD_GETDPI(windowid, dpix, dpiy)

        IMPLICIT NONE
        INTEGER windowid
        REAL dpix, dpiy

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GETDPI: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GETDPI: null windowobj'
        ENDIF

        CALL FGDWINDPI(success, windowobjs(windowid), dpix, dpiy)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

