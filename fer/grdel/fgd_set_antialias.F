*
* Turns on or off anti-aliasing on items drawn after this call
* in all existing and future windows.
*
* Input Arguments:
*     aaint - if non-zero, turn on antialiasing;
*             if zero, turn off antialiasing
*
* TODO: when SET WIN /ANTIALIAS implemented, have this function
*       only set antialiasing for future windows.
*
        SUBROUTINE FGD_SET_ANTIALIAS(aaint)

        IMPLICIT NONE
        INTEGER aaint

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Local variables
        INTEGER k, success, errmsglen
        CHARACTER*2048 errmsg

*       Inform all existing windows
*       To be eliminated when SET WIN /ANTIALIAS implemented
        DO 10 k = 1, maxwindowobjs
            IF ( windowobjs(k) .NE. nullobj ) THEN
                CALL FGDWINSETANTIALIAS(success, windowobjs(k), aaint)
                IF ( success .EQ. 0 ) THEN
                    errmsg = ' '
                    CALL FGDERRMSG(errmsg, errmsglen)
                    CALL SPLIT_LIST(pttmode_help, err_lun,
     .                              errmsg, errmsglen)
                ENDIF
            ENDIF
   10   CONTINUE

*       Set the default for future windows
        IF ( aaint .NE. 0 ) THEN
           useantialiasing = .TRUE.
        ELSE
           useantialiasing = .FALSE.
        ENDIF

        RETURN
        END

