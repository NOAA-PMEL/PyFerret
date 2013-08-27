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
        INCLUDE 'xplot_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Function return value definition
        INTEGER TM_LENSTR

*       Local variables
        INTEGER fnamelen, fmtlen, transpbkg, success, errmsglen
        REAL*4 xinches, yinches
        INTEGER xpixels, ypixels
        CHARACTER*8 fformat
        CHARACTER*2048 errmsg
        REAL*8 memory(2)
        INTEGER firststr, numstrs

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
        xinches = wn_xinches(activewindow)
        yinches = wn_yinches(activewindow)
        xpixels = NINT(wn_xpixels(activewindow))
        ypixels = NINT(wn_ypixels(activewindow))
*       Just some dummy variables to indicate no annotations
        memory(1) = 0
        memory(2) = 0
        firststr = 0
        numstrs = 0
        CALL FGDWINSAVE(success, windowobjs(activewindow), filename,
     .                  fnamelen, fformat, fmtlen, transpbkg,
     .                  xinches, yinches, xpixels, ypixels,
     .                  memory, firststr, numstrs)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

