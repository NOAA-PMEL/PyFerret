*
* Overlays specified image as a watermark
*
* Input Arguments:
*     windowid - number of the window to use
*     filename - water mark image filename
*     len_filename - length of image filename

        SUBROUTINE FGD_SET_WMARK(windowid, filename, len_filename)

        IMPLICIT NONE
        INTEGER windowid
        CHARACTER*2048 filename
        INTEGER len_filename

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Local variables
        INTEGER success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_WMARK: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       This may be called prior to window creation
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
*           Update the width scaling factor of this window
            CALL FGDWINSETWMARK(success,
     .                          windowobjs(windowid), filename, len_filename)
            IF ( success .EQ. 0 ) THEN
*               Error - output error message
                errstr = ' '
                CALL FGDERRMSG(errstr, errstrlen)
                CALL SPLIT_LIST(pttmode_help,
     .                          err_lun, errstr, errstrlen)
                RETURN
            ENDIF
        ENDIF

        RETURN
        END
