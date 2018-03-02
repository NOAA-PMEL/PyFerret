*
* Set clipping indicator for the active window.
*

        SUBROUTINE FGD_GSCLIP(clipit)

        IMPLICIT NONE
        INTEGER clipit

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSCLIP: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSCLIP: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
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
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

        RETURN
        END

