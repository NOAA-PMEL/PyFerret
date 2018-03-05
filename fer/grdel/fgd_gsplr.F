*
* Create a pen object.  (set polyline representation)
*
* The linewidth in GSK was a factor, so not very clear
* what the units should be.  The line widths in PyFerret
* are in units of points (1/72 inches); possibly further 
* scaled by the line width scaling factor.
*

        SUBROUTINE FGD_GSPLR(windowid, pennum, linestyle,
     .                       linewidth, colorindex)

        IMPLICIT NONE
        INTEGER windowid, pennum, linestyle, colorindex
        REAL*4 linewidth

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, stylen, cstylen, jstylen,
     .          success, errstrlen, TM_LENSTR
        CHARACTER*64 stylename, capstyle, joinstyle
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSPLR: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPLR: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (pennum .LT. 1) .OR. (pennum .GT. maxpenobjs) ) THEN
            errstr = 'FGD_GSPLR: invalid pennum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (colorindex .LT. 0) .OR.
     .       (colorindex .GE. maxcolorobjs) ) THEN
            errstr = 'FGD_GSPLR: invalid colorindex'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        colornum = colorindex + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPLR: null colorobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
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
            errstr = 'FGD_GSPLR: invalid linestyle'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        capstyle = 'square'
        cstylen = 6
        joinstyle = 'bevel'
        jstylen = 5

*       If a pen object already exists, delete it
        IF ( penobjs(pennum, windowid) .NE. nullobj ) THEN
            CALL FGDPENDEL(success, penobjs(pennum, windowid))
            penobjs(pennum, windowid) = nullobj
        ENDIF

*       Create the new pen object
        CALL FGDPEN(penobjs(pennum, windowid), windowobjs(windowid),
     .          colorobjs(colornum, windowid), linewidth, stylename,
     .          stylen, capstyle, cstylen, joinstyle, jstylen)
        IF ( penobjs(pennum, windowid) .EQ. nullobj ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        pencolor(pennum, windowid) = colornum
        penwidth(pennum, windowid) = linewidth

        RETURN
        END
