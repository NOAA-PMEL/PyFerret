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

        INTEGER k, success, errmsglen
        CHARACTER*2048 errmsg
        REAL*8 tempbrush, fillbrush
        REAL*4 left, bottom, right, top
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
            tempbrush = nullobj
        ELSE
*           Create a temporary brush to use
            IF ( (lastbrushcolor .LT. 1) .OR.
     .           (lastbrushcolor .GT. maxcolorobjs) ) THEN
                STOP 'FGD_GFA: invalid lastbrushcolor'
            ENDIF
            IF ( colorobjs(lastbrushcolor, activewindow)
     .           .EQ. nullobj ) THEN
                STOP 'FGD_GFA: null lastbrushcolor'
            ENDIF
            CALL FGDBRUSH(tempbrush, windowobjs(activewindow),
     .                    colorobjs(lastbrushcolor, activewindow),
     .                    lastbrushstyle, lastbstylen)
            IF ( tempbrush .EQ. nullobj ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
                STOP 'FGD_GFA: null tempbrush'
            ENDIF
            fillbrush = tempbrush
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
     .                  left, bottom, right, top, fillbrush, nullobj)
                rectdrawn = .TRUE.
            ENDIF
        ENDIF
        IF ( .NOT. rectdrawn ) THEN
*           Fill generic polygon area (no outline)
            CALL FGDDRAWPOLYGON(success, windowobjs(activewindow),
     .                          ptsx, ptsy, npts, fillbrush, nullobj)
        ENDIF
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        somethingdrawn = .TRUE.

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

