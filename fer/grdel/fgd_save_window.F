*
* Save the active window graphics to file.
* This is intended to be used as the alternative
* of writing a GKS metafile by PlotPlus.
*
* Input Arguments:
*     filename: save the image under this filename;
*               image format guessed from the filename extension
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
        INTEGER fnamelen, fmtlen, transpbkg, success, errstrlen
        REAL*4 xinches, yinches
        INTEGER xpixels, ypixels
        CHARACTER*8 fformat
        CHARACTER*2048 errstr
        REAL*8 memory(2)
        INTEGER firststr, numstrs

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SAVE_WINDOW: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_SAVE_WINDOW: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
*       Get the length of the filename
        fnamelen = TM_LENSTR(filename)
        IF ( fnamelen .LE. 0 ) THEN
            errstr = 'FGD_SAVE_WINDOW: blank filename'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
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
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

        RETURN
        END

