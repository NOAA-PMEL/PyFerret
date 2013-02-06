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
        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SET_ANTIALIAS: Invalid windowid value'
        ENDIF

*       This may be called to prior to window creation
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
*           Update the anti-aliasing state of this window
            CALL FGDWINSETANTIALIAS(success, 
     .                              windowobjs(windowid), aaint)
            IF ( success .EQ. 0 ) THEN
*               Error - output error message
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, 
     .                          err_lun, errmsg, errmsglen)
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

