*
* Save the active window graphics to file.
* This is intended to be used as the alternative
* of writing a GKS metafile by PlotPlus.
*
* Input Arguments:
*     windowid: ID number of the Window to be closed
*
        SUBROUTINE FGD_SAVE_WINDOW(filename)

        IMPLICIT NONE
        CHARACTER*(*) filename

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Function return value definition
        INTEGER TM_LENSTR

*       Local variables
        INTEGER fnamelen, fmtlen, transpbkg, success, errmsglen
        CHARACTER*8 fformat
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SAVE_WINDOW: Invalid activewindow value'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_SAVE_WINDOW: null windowobj'
        ENDIF
*       Get the length of the filename
        fnamelen = TM_LENSTR(filename)
        IF ( fnamelen .LE. 0 ) THEN
            STOP 'FGD_SAVE_WINDOW: empty filename'
        ENDIF
*       Let it guess the format from the filename extension
        fformat = ' '
        fmtlen = 0
*       Set the background transparency from the common block value
        IF ( savetransparent ) THEN
            transpbkg = 1
        ELSE
            transpbkg = 0
        ENDIF
*       Save the graphics to file
        CALL FGDWINSAVE(success, windowobjs(activewindow), filename,
     .                  fnamelen, fformat, fmtlen, transpbkg)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

