*
* Turns on or off anti-aliasing on items drawn after this call
* in the specified window
*
* Input Arguments:
*     windowid - number of the window to use
*     aaint - if non-zero, turn on antialiasing;
*             if zero, turn off antialiasing
*
        SUBROUTINE FGD_SET_ANTIALIAS(windowid, aaint)

        IMPLICIT NONE
        INTEGER windowid, aaint

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Local variables
        INTEGER success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_ANTIALIAS: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       This may be called to prior to window creation
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
*           Update the anti-aliasing state of this window
            CALL FGDWINSETANTIALIAS(success, 
     .                              windowobjs(windowid), aaint)
            IF ( success .EQ. 0 ) THEN
*               Error - output error message
                errstr = ' '
                CALL FGDERRMSG(errstr, errstrlen)
                CALL SPLIT_LIST(pttmode_help, 
     .                          err_lun, errstr, errstrlen)
                RETURN
            ENDIF
        ENDIF

*       Record the current anti-aliasing state
        IF ( aaint .EQ. 0 ) THEN
            antialias(windowid) = .FALSE.
        ELSE
            antialias(windowid) = .TRUE.
        ENDIF

        RETURN
        END

