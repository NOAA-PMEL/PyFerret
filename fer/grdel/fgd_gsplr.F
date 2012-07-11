*
* Create a pen object.  (set polyline representation)
*
* The linewidth in GSK was a factor, so not very clear
* what the units should be.  The line widths here scale
* with the size of the view, but still not sure what
* the units actually are.  The factor applied to the
* line width was determined to reproduce Ferret line
* widths in a default screen.
*

        SUBROUTINE FGD_GSPLR(windowid, pennum, linestyle,
     .                       linewidth, colorindex)

        IMPLICIT NONE
        INTEGER windowid, pennum, linestyle, colorindex
        REAL linewidth

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, stylen, cstylen, jstylen,
     .          success, errmsglen
        CHARACTER*64 stylename, capstyle, joinstyle
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSPLR: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSPLR: null windowobj'
        ENDIF
        IF ( (pennum .LT. 1) .OR. (pennum .GT. maxpenobjs) ) THEN
            STOP 'FGD_GSPLR: Invalid pennum'
        ENDIF
        IF ( (colorindex .LT. 0) .OR.
     .       (colorindex .GE. maxcolorobjs) ) THEN
            STOP 'FGD_GSPLR: Invalid colorindex'
        ENDIF

        colornum = colorindex + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSPLR: null colorobj'
        ENDIF

*       Line type
        IF ( linestyle .EQ. 1 ) THEN
            stylename = 'solid'
            stylen = 5
        ELSE IF ( linestyle .EQ. 2 ) THEN
            stylename = 'dash'
            stylen = 4
        ELSE IF ( linestyle .EQ. 3 ) THEN
            stylename = 'dot'
            stylen = 3
        ELSE IF ( linestyle .EQ. 4 ) THEN
            stylename = 'dashdot'
            stylen = 7
        ELSE
            STOP 'FGD_GSPLR: invalid linestyle'
        ENDIF
        capstyle = 'square'
        cstylen = 6
        joinstyle = 'bevel'
        jstylen = 5

*       If a pen object already exists, delete it
        IF ( penobjs(pennum, windowid) .NE. nullobj ) THEN
            CALL FGDPENDEL(success, penobjs(pennum, windowid))
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
            penobjs(pennum, windowid) = nullobj
        ENDIF

*       Create the new pen object
        CALL FGDPEN(penobjs(pennum, windowid), windowobjs(windowid),
     .          colorobjs(colornum, windowid), linewidth, stylename,
     .          stylen, capstyle, cstylen, joinstyle, jstylen)
        IF ( penobjs(pennum, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        pencolor(pennum, windowid) = colornum

        RETURN
        END
