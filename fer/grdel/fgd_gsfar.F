*
* Creates a Brush object.
* (set fill area representation)
*

        SUBROUTINE FGD_GSFAR(windowid, brushnum,
     .                       style, hstyle, colorndx)

        IMPLICIT NONE
        INTEGER windowid, brushnum, style, hstyle, colorndx

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errmsglen, bstylen
        CHARACTER*2048 errmsg
        CHARACTER*64 brushstyle

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSFAR: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSFAR: null windowobj'
        ENDIF
        IF ( (brushnum .LT. 1) .OR.
     .       (brushnum .GT. maxbrushobjs) ) THEN
            STOP 'FGD_GSFAR: Invalid brushnum'
        ENDIF
        IF ( (colorndx .LT. 0) .OR.
     .       (colorndx .GE. maxcolorobjs) ) THEN
            STOP 'FGD_GSFAR: Invalid colorndx'
        ENDIF
        colornum = colorndx + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSFAR: null colorobj'
        ENDIF

        IF ( style .EQ. 1 ) THEN
*           Solid brush
            brushstyle = "solid"
            bstylen = 5
        ELSE IF ( style .EQ. 3 ) THEN
*           Hatch brush styles
            IF ( hstyle .EQ. 1 ) THEN
                brushstyle = "hor"
                bstylen = 3
            ELSE IF ( hstyle .EQ. 2 ) THEN
                brushstyle = "fdiag"
                bstylen = 5
            ELSE IF ( hstyle .EQ. 3 ) THEN
                brushstyle = "ver"
                bstylen = 3
            ELSE IF ( hstyle .EQ. 4 ) THEN
                brushstyle = "bdiag"
                bstylen = 5
            ELSE IF ( hstyle .EQ. 5 ) THEN
                brushstyle = "cross"
                bstylen = 5
            ELSE IF ( hstyle .EQ. 6 ) THEN
                brushstyle = "diagcross"
                bstylen = 9
            ELSE
                STOP 'FGD_GSFAR: Invalid hstyle'
            ENDIF
        ELSE
            STOP 'FGD_GSFAR: Invalid style'
        ENDIF

*       If a brush object already exists, delete it
        IF ( brushobjs(brushnum, windowid) .NE. nullobj ) THEN
            CALL FGDBRUSHDEL(success, brushobjs(brushnum, windowid))
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
            brushobjs(brushnum, windowid) = nullobj
        ENDIF

*       Create the new brush object
        CALL FGDBRUSH(brushobjs(brushnum, windowid),
     .                windowobjs(windowid),
     .                colorobjs(colornum, windowid),
     .                brushstyle, bstylen)

        IF ( brushobjs(brushnum, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        brushcolor(brushnum, windowid) = colornum

        RETURN
        END

