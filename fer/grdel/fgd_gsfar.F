*
* Creates a Brush object.
* (set fill area representation)
*

        SUBROUTINE FGD_GSFAR(windowid, brushnum,
     .                       style, hstyle, colorndx)

        IMPLICIT NONE
        INTEGER windowid, brushnum, style, hstyle, colorndx

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errstrlen, bstylen, TM_LENSTR
        CHARACTER*2048 errstr
        CHARACTER*64 brushstyle

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSFAR: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSFAR: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (brushnum .LT. 1) .OR.
     .       (brushnum .GT. maxbrushobjs) ) THEN
            errstr = 'FGD_GSFAR: invalid brushnum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (colorndx .LT. 0) .OR.
     .       (colorndx .GE. maxcolorobjs) ) THEN
            errstr = 'FGD_GSFAR: invalid colorndx'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        colornum = colorndx + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSFAR: null colorobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
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
                errstr = 'FGD_GSFAR: invalid hstyle'
                errstrlen = TM_LENSTR(errstr)
                CALL SPLIT_LIST(pttmode_help,err_lun,errstr,errstrlen)
                RETURN
            ENDIF
        ELSE
            errstr = 'FGD_GSFAR: invalid style'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       If a brush object already exists, delete it
        IF ( brushobjs(brushnum, windowid) .NE. nullobj ) THEN
            CALL FGDBRUSHDEL(success, brushobjs(brushnum, windowid))
            brushobjs(brushnum, windowid) = nullobj
        ENDIF

*       Create the new brush object
        CALL FGDBRUSH(brushobjs(brushnum, windowid),
     .                windowobjs(windowid),
     .                colorobjs(colornum, windowid),
     .                brushstyle, bstylen)

        IF ( brushobjs(brushnum, windowid) .EQ. nullobj ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        brushcolor(brushnum, windowid) = colornum

        RETURN
        END

