*
* Definition for GDAWK (deactivate workstation)
* Ends the view of the active window (if valid)
* and zeros out activewindow
*

        SUBROUTINE FGD_GDAWK(wk_id)

        IMPLICIT NONE
        INTEGER wk_id

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, errstrlen
        CHARACTER*2048 errstr

*       End the view of a valid active window
        IF ( (activewindow .GE. 1) .AND.
     .       (activewindow .LE. maxwindowobjs) ) THEN
            CALL FGDVIEWEND(success, windowobjs(activewindow))
            IF ( success .EQ. 0 ) THEN
                errstr = ' '
                CALL FGDERRMSG(errstr, errstrlen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errstr, errstrlen)
            ENDIF
        ENDIF

*       set so there are no active windows
        activewindow = 0

        RETURN
        END

