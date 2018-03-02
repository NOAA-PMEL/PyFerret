*
* Draw points in the active window using
* the active marker object (polymarker)
*

        SUBROUTINE FGD_GPM(npts, ptsx, ptsy)

        IMPLICIT NONE
        INTEGER npts
        REAL*4 ptsx, ptsy
        DIMENSION ptsx(npts), ptsy(npts)

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL*8  symbolobj, colorobj
        REAL*4  marksize
        INTEGER symbolnum, colornum, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GPM: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GPM: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (activemarker .LT. 1) .OR.
     .       (activemarker .GT. maxmarkers) ) THEN
            errstr = 'FGD_GPM: no active marker'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
            success = 0
        symbolnum = markersymbol(activemarker, activewindow)
        IF ( (symbolnum .LT. 1) .OR. 
     .       (symbolnum .GT. maxsymbolobjs) ) THEN
            errstr = 'FGD_GPM: invalid marker symbol number'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        symbolobj = symbolobjs(symbolnum, activewindow)
        IF ( symbolobj .EQ. nullobj ) THEN
            errstr = 'FGD_GPM: invalid marker symbol'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        colornum = markercolor(activemarker, activewindow)
        IF ( (colornum .LT. 1) .OR. 
     .       (colornum .GT. maxcolorobjs) ) THEN
            errstr = 'FGD_GPM: no marker color'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        colorobj = colorobjs(colornum, activewindow)
        IF ( colorobj .EQ. nullobj ) THEN
            errstr = 'FGD_GPM: invalid marker color'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        marksize = markersize(activemarker, activewindow)
        IF ( marksize .LE. 0.0 ) THEN
            errstr = 'FGD_GPM: invalid marker size'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Draw the markers
        CALL FGDDRAWPOINTS(success, windowobjs(activewindow),
     .                     ptsx, ptsy, npts,
     .                     symbolobj, colorobj, marksize)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        somethingdrawn = .TRUE.

        RETURN
        END

