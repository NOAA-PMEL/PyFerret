*
* Set clipping indicator for the active window.
*

        SUBROUTINE FGD_GSCLIP(clipit)

        IMPLICIT NONE
        INTEGER clipit

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSCLIP: Invalid activewindow value'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GSCLIP: null activewindow'
        ENDIF
*       Record the clipping state
        IF ( clipit .EQ. 0 ) THEN
            cliptoview(activewindow) = .FALSE.
        ELSE
            cliptoview(activewindow) = .TRUE.
        ENDIF
*       Assign the clipping state
        CALL FGDVIEWCLIP(success, windowobjs(activewindow), clipit)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

