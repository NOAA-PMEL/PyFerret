*
* Turns on or off anti-aliasing on items drawn after this call
* in all existing and future windows.
*
* Input Arguments:
*     antialias - turn on antialiasing?
*
        SUBROUTINE FGD_SET_ANTIALIAS(antialias)

        IMPLICIT NONE
        LOGICAL antialias

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Local variables
        INTEGER k, aaint, success, errmsglen
        CHARACTER*2048 errmsg

*       Inform all existing windows
        IF ( antialias ) THEN
            aaint = 1
        ELSE
            aaint = 0
        ENDIF
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
        useantialiasing = antialias
        RETURN
        END

