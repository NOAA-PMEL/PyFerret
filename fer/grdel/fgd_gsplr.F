*
* Create a pen object.
* (set polyline representation)
*

        SUBROUTINE FGD_GSPLR(windowid, pennum, 
     .                 linestyle, linewidth, colorindex)

        IMPLICIT NONE
        INTEGER windowid, pennum, linestyle, colorindex
        REAL linewidth

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, stylen, cstylen, jstylen,
     .          success, errmsglen
        CHARACTER*64 stylename, capstyle, joinstyle
        REAL adjwidth
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
        capstyle = 'round'
        cstylen = 5
        joinstyle = 'round'
        jstylen = 5

*       Adjust the line width to units of 0.001 of 
*       the longest side of the current View
        adjwidth = 16.0 * linewidth

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
     .          colorobjs(colornum, windowid), adjwidth, stylename,
     .          stylen, capstyle, cstylen, joinstyle, jstylen)
        IF ( penobjs(pennum, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END
