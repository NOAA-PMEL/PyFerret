*
* Return the size of the active window in pixels.
*

        SUBROUTINE FGD_GET_WINDOW_SIZE(width, height)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL*4 width, height

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GET_WINDOW_SIZE: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GET_WINDOW_SIZE: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        width = windowwidth(activewindow) *
     .          windowdpix(activewindow)
        height = windowheight(activewindow) *
     .           windowdpiy(activewindow)

        RETURN
        END

