*
* Create a symbol object.
* (set polymarker representation)
*
* The marker size in GSK was a factor, so not very clear
* what the units should be.  The marker size in PyFerret
* are in units of points (1/72 inches); possibly further 
* scaled by the line width scaling factor.
*

        SUBROUTINE FGD_GSPMR(windowid, marknum,
     .                       symnum, symsize, colorindex)

        IMPLICIT NONE
        INTEGER windowid, marknum, symnum, colorindex
        REAL*4 symsize

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER TM_LENSTR

        INTEGER colornum, success, errstrlen
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSPMR: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPMR: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (marknum .LT. 1) .OR.
     .       (marknum .GT. maxmarkers) ) THEN
            errstr = 'FGD_GSPMR: invalid marknum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (symnum .LT. 1) .OR.
     .       (symnum .GT. maxsymbolobjs) ) THEN
            errstr = 'FGD_GSPMR: invalid symnum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( symbolobjs(symnum, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPMR: null symbolobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        colornum = colorindex + 1
        IF ( (colornum .LT. 1) .OR.
     .       (colornum .GT. maxcolorobjs) ) THEN
            errstr = 'FGD_GSPMR: invalid colorindex'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPMR: null colorobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( symsize .LE. 0.0 ) THEN
            errstr = 'FGD_GSPMR: invalid symsize'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        markersymbol(marknum, windowid) = symnum
        markercolor(marknum, windowid) = colornum
        markersize(marknum, windowid) = symsize

        RETURN
        END
