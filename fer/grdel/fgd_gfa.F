*
* Fill an area in the active window using the active brush object
* or a temporary brush created from the last brush description.
*

        SUBROUTINE FGD_GFA(npts, ptsx, ptsy)

        IMPLICIT NONE
        INTEGER npts
        REAL*4 ptsx, ptsy
        DIMENSION ptsx(npts), ptsy(npts)

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER k, success, errmsglen, stylen, cstylen, jstylen
        CHARACTER*64 style, capstyle, joinstyle
        CHARACTER*2048 errmsg
        REAL*8 brushcolorobj, tempbrush, fillbrush, outlinepen
        REAL*4 left, bottom, right, top, linewidth
        LOGICAL rectdrawn;

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GFA: No active window'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GFA: null active windowobj'
        ENDIF

        IF ( (activebrush .GE. 1) .AND.
     .       (activebrush .LE. maxbrushobjs) ) THEN
*           Use the predefined brush
            fillbrush = brushobjs(activebrush, activewindow)
            IF ( fillbrush .EQ. nullobj ) THEN
                STOP 'FGD_GFA: null activebrush'
            ENDIF
*           Get the color object used by this brush in case we need it
            k = brushcolor(activebrush, activewindow)
            IF ( (k .LT. 1) .OR. (k .GT. maxcolorobjs) ) THEN
                STOP 'FGD_GFA: invalid brushcolor for active brush'
            ENDIF
            brushcolorobj = colorobjs(k,activewindow)
            IF ( brushcolorobj .EQ. nullobj ) THEN
                STOP 'FGD_GFA: null brushcolor for active brush'
            ENDIF
            tempbrush = nullobj
        ELSE
*           Create a temporary brush to use
            IF ( (lastbrushcolor .LT. 1) .OR.
     .           (lastbrushcolor .GT. maxcolorobjs) ) THEN
                STOP 'FGD_GFA: invalid lastbrushcolor'
            ENDIF
            brushcolorobj = colorobjs(lastbrushcolor, activewindow)
            IF ( brushcolorobj .EQ. nullobj ) THEN
                STOP 'FGD_GFA: null lastbrushcolor'
            ENDIF
            CALL FGDBRUSH(tempbrush, windowobjs(activewindow),
     .                    brushcolorobj, lastbrushstyle, lastbstylen)
            IF ( tempbrush .EQ. nullobj ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
                STOP 'FGD_GFA: null tempbrush'
            ENDIF
            fillbrush = tempbrush
        ENDIF

        IF ( outlinewidth(activewindow) .GT. 0.0 ) THEN
            linewidth = outlinewidth(activewindow) *
     .                  thickfactor(activewindow)
            style = "solid"
            stylen = 5
            capstyle = 'square'
            cstylen = 6
            joinstyle = 'bevel'
            jstylen = 5
            CALL FGDPEN(outlinepen, windowobjs(activewindow),
     .                  brushcolorobj, linewidth, style, stylen,
     .                  capstyle, cstylen, joinstyle, jstylen)
        ELSE
            outlinepen = nullobj
        ENDIF

*       Quick check for a rectangle as specified by Ferret
*       (Drawing rectangles much faster than generic polygons.)
        rectdrawn = .FALSE.
        IF ( npts .EQ. 4 ) THEN
            IF ( (ptsx(1) .EQ. ptsx(2)) .AND.
     .           (ptsx(3) .EQ. ptsx(4)) .AND.
     .           (ptsy(1) .EQ. ptsy(4)) .AND.
     .           (ptsy(2) .EQ. ptsy(3)) ) THEN
                IF ( ptsx(1) .LT. ptsx(3) ) THEN
                    left = ptsx(1)
                    right = ptsx(3)
                ELSE
                    left = ptsx(3)
                    right = ptsx(1)
                ENDIF
                IF ( ptsy(1) .LT. ptsy(2) ) THEN
                    bottom = ptsy(1)
                    top = ptsy(2)
                ELSE
                    bottom = ptsy(2)
                    top = ptsy(1)
                ENDIF
*               Fill rectangle area (no outline)
                CALL FGDDRAWRECT(success, windowobjs(activewindow),
     .                  left, bottom, right, top, fillbrush, outlinepen)
                rectdrawn = .TRUE.
            ENDIF
        ENDIF
        IF ( .NOT. rectdrawn ) THEN
*           Fill generic polygon area (no outline)
            CALL FGDDRAWPOLYGON(success, windowobjs(activewindow),
     .                          ptsx, ptsy, npts, fillbrush, outlinepen)
        ENDIF
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        somethingdrawn = .TRUE.

*       If an outline pen was created, delete it
        IF ( outlinepen .NE. nullobj ) THEN
            CALL FGDPENDEL(success, outlinepen)
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
        ENDIF

*       If a temporary brush was created, delete it
        IF ( tempbrush .NE. nullobj ) THEN
            CALL FGDBRUSHDEL(success, tempbrush)
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
        ENDIF

        RETURN
        END

