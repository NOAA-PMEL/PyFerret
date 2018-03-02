*
* Create color objects for a Window.
*

        SUBROUTINE FGD_GSCR(windowid, colorindex,
     .             redfrac, greenfrac, bluefrac, opaquefrac)

        IMPLICIT NONE
        INTEGER windowid, colorindex
        REAL*4 redfrac, greenfrac, bluefrac, opaquefrac

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr
        INTEGER brushnum, pennum

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSCR: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSCR: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (colorindex .LT. 0) .OR.
     .       (colorindex .GE. maxcolorobjs) ) THEN
            errstr = 'FGD_GSCR: invalid colorindex'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (redfrac .LT. 0.0) .OR.  (redfrac .GT. 1.0) ) THEN
            errstr = 'FGD_GSCR: invalid redfrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (greenfrac .LT. 0.0) .OR.  (greenfrac .GT. 1.0) ) THEN
            errstr = 'FGD_GSCR: invalid greenfrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (bluefrac .LT. 0.0) .OR.  (bluefrac .GT. 1.0) ) THEN
            errstr = 'FGD_GSCR: invalid bluefrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (opaquefrac .LT. 0.0) .OR.  (opaquefrac .GT. 1.0) ) THEN
            errstr = 'FGD_GSCR: invalid opaquefrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        colornum = colorindex + 1

*       If this is replacing an existing color object,
*       delete the previous color object.
        IF ( colorobjs(colornum, windowid) .NE. nullobj ) THEN
            CALL FGDCOLORDEL(success, colorobjs(colornum, windowid))
            IF ( success .EQ. 0 ) THEN
                errstr = ' '
                CALL FGDERRMSG(errstr, errstrlen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errstr, errstrlen)
                RETURN
            ENDIF
            colorobjs(colornum, windowid) = nullobj
        ENDIF

*       Create the new color object.
        CALL FGDCOLOR(colorobjs(colornum, windowid),
     .                windowobjs(windowid),
     .                redfrac, greenfrac, bluefrac, opaquefrac)
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Simplest approach for retrieving color RGBA values was 
*       just to store them in the common block when they are set
        colorrgba(1, colornum, windowid) = redfrac
        colorrgba(2, colornum, windowid) = greenfrac
        colorrgba(3, colornum, windowid) = bluefrac
        colorrgba(4, colornum, windowid) = opaquefrac

*       Update any brushes that used this color number
        DO 10 brushnum = 1, maxbrushobjs
            IF ( (brushobjs(brushnum, windowid)  .NE. nullobj) .AND.
     .           (brushcolor(brushnum, windowid) .EQ. colornum) ) THEN
                CALL FGDBRUSHREPLACECOLOR(success,
     .                       brushobjs(brushnum, windowid),
     .                       colorobjs(colornum, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun,
     .                              errstr, errstrlen)
                    RETURN
                ENDIF
            ENDIF
   10   CONTINUE

*       Update any pens that used this color number
        DO 20 pennum = 1, maxpenobjs
            IF ( (penobjs(pennum, windowid)  .NE. nullobj) .AND.
     .           (pencolor(pennum, windowid) .EQ. colornum) ) THEN
                CALL FGDPENREPLACECOLOR(success,
     .                     penobjs(pennum, windowid),
     .                     colorobjs(colornum, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun,
     .                              errstr, errstrlen)
                    RETURN
                ENDIF
            ENDIF
   20   CONTINUE

        RETURN
        END

