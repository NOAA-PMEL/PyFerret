*
* Definition for GDAWK (deactivate workstation)
* Ends the view of the active window (if valid)
* and zeros out activewindow
*

        SUBROUTINE FGD_GDAWK(wk_id)

        IMPLICIT NONE
        INTEGER wk_id

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       End the view of a valid active window
        IF ( (activewindow .GE. 1) .AND.
     .       (activewindow .LE. maxwindowobjs) ) THEN
            CALL FGDVIEWEND(success, windowobjs(activewindow))
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
        ENDIF

*       set so there are no active windows
        activewindow = 0

        RETURN
        END

