*
* Inquire clipping indicator for the active window.
*

        SUBROUTINE FGD_GQCLIP(stat, clipit)

        IMPLICIT NONE
        INTEGER stat, clipit

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GQCLIP: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            stat = 1
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GQCLIP: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            stat = 1
            RETURN
        ENDIF
*       Inquire the clipping state
        IF ( cliptoview(activewindow) ) THEN
            clipit = 1
        ELSE
            clipit = 0
        ENDIF
        stat = 0

        RETURN
        END

