*
* Create color objects for a Window.
*

        SUBROUTINE FGD_GSCR(windowid, colorindex,
     .                  redfrac, greenfrac, bluefrac)

        IMPLICIT NONE
        INTEGER windowid, colorindex
        REAL*4 redfrac, greenfrac, bluefrac

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, success, errmsglen
        REAL*4 opaquefrac
        CHARACTER*2048 errmsg
        INTEGER brushnum, pennum

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSCR: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSCR: null windowobj'
        ENDIF
        IF ( (colorindex .LT. 0) .OR.
     .       (colorindex .GE. maxcolorobjs) ) THEN
            STOP 'FGD_GSCR: Invalid colorindex'
        ENDIF
        IF ( (redfrac .LT. 0.0) .OR.  (redfrac .GT. 1.0) ) THEN
            STOP 'FGD_GSCR: Invalid redfrac'
        ENDIF
        IF ( (greenfrac .LT. 0.0) .OR.  (greenfrac .GT. 1.0) ) THEN
            STOP 'FGD_GSCR: Invalid greenfrac'
        ENDIF
        IF ( (bluefrac .LT. 0.0) .OR.  (bluefrac .GT. 1.0) ) THEN
            STOP 'FGD_GSCR: Invalid bluefrac'
        ENDIF

        colornum = colorindex + 1

*       If this is replacing an existing color object,
*       delete the previous color object.
        IF ( colorobjs(colornum, windowid) .NE. nullobj ) THEN
            CALL FGDCOLORDEL(success, colorobjs(colornum, windowid))
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
            colorobjs(colornum, windowid) = nullobj
        ENDIF

*       Create the new color object.
*       Hard coded to be opaque for now.
        opaquefrac = 1.0
        CALL FGDCOLOR(colorobjs(colornum, windowid),
     .                windowobjs(windowid),
     .                redfrac, greenfrac, bluefrac, opaquefrac)
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        colorrgb(1, colornum, windowid) = redfrac
        colorrgb(2, colornum, windowid) = greenfrac
        colorrgb(3, colornum, windowid) = bluefrac

*       Update any brushes that used this color number
        DO 10 brushnum = 1, maxbrushobjs
            IF ( (brushobjs(brushnum, windowid)  .NE. nullobj) .AND.
     .           (brushcolor(brushnum, windowid) .EQ. colornum) ) THEN
                CALL FGDBRUSHREPLACECOLOR(success,
     .                       brushobjs(brushnum, windowid),
     .                       colorobjs(colornum, windowid))
                IF ( success .EQ. 0 ) THEN
                    errmsg = ' '
                    CALL FGDERRMSG(errmsg, errmsglen)
                    CALL SPLIT_LIST(pttmode_help, err_lun,
     .                              errmsg, errmsglen)
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
                    errmsg = ' '
                    CALL FGDERRMSG(errmsg, errmsglen)
                    CALL SPLIT_LIST(pttmode_help, err_lun,
     .                              errmsg, errmsglen)
                ENDIF
            ENDIF
   20   CONTINUE

        RETURN
        END

