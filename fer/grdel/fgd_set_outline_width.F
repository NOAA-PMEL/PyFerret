*
* Assigns the width of outlines for rectangles and other polygons
* in the specified window
*
* Input Arguments:
*     windowid - number of the window to use
*     linewidth - width of outlines; can be zero for no outlines
*
        SUBROUTINE FGD_SET_OUTLINE_WIDTH(windowid, linewidth)

        IMPLICIT NONE
        INTEGER windowid
        REAL*4 linewidth

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'


        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_OUTLINE_WIDTH: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( linewidth .LT. 0.0 ) THEN
            errstr = 'FGD_SET_OUTLINE_WIDTH: invalid linewidth'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Record the outline width for this window
        outlinewidth(windowid) = linewidth

        RETURN
        END

