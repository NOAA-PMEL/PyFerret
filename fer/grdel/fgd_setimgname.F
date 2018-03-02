*
* Informs the currently active window the filename (and format, dervied
* from the filename entesion) for the image.  This may just be a default
* filename when saving a window (for interactive graphics window), or it
* may be the filename which is written as the drawing proceeds (for non-
* interactive "batch mode" graphics without a display window).
*
* Input arguments:
*    imgname: filename for the image
*

        SUBROUTINE FGD_SETIMGNAME(imgname, namesent)

        IMPLICIT NONE
        CHARACTER*(*) imgname
        LOGICAL namesent

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER TM_LENSTR

        INTEGER imgnamelen, fmtnamelen, success, errstrlen
        CHARACTER*8 fmtname
        CHARACTER*2048 errstr

*       This could be called before creating a window
        namesent = .FALSE.
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            RETURN
        ENDIF

        imgnamelen = TM_LENSTR(imgname)
        fmtname = ' '
        fmtnamelen = 0
        CALL FGDWINIMGNAME(success, windowobjs(activewindow),
     .                     imgname, imgnamelen, fmtname, fmtnamelen)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        namesent = .TRUE.

        RETURN
        END

