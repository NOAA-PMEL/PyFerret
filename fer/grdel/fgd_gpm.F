*
* Draw points in the active window using
* the active symbol object (polymarker)
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

        INTEGER colornum, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GPM: Invalid activewindow'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GPM: null activewindow'
        ENDIF
        IF ( (activesymbol .LT. 1) .OR.
     .       (activesymbol .GT. maxsymbolobjs) ) THEN
            STOP 'FGD_GPM: Invalid activesymbol'
        ENDIF
        IF ( symbolobjs(activesymbol, activewindow)
     .       .EQ. nullobj ) THEN
            STOP 'FGD_GPM: null activesymbol'
        ENDIF
        colornum = symbolcolor(activesymbol, activewindow)
        IF ( (colornum .LT. 1) .OR.
     .       (colornum .GT. maxcolorobjs) ) THEN
            STOP 'FGD_GPM: Invalid symbolcolor'
        ENDIF
        IF ( colorobjs(colornum, activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GPM: null symbolcolor'
        ENDIF
        IF ( symbolsize(activesymbol, activewindow) .LE. 0.0 ) THEN
            STOP 'FGD_GPM: invalid symbolsize'
        ENDIF

*       Draw the symbols
        CALL FGDDRAWPOINTS(success, windowobjs(activewindow),
     .                     ptsx, ptsy, npts,
     .                     symbolobjs(activesymbol, activewindow),
     .                     colorobjs(colornum, activewindow),
     .                     symbolsize(activesymbol, activewindow))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        somethingdrawn = .TRUE.

        RETURN
        END

