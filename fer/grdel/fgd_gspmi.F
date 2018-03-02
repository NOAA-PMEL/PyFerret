*
* Set the active marker of the active window
* (set polymarker index)
*

        SUBROUTINE FGD_GSPMI(marknum)

        IMPLICIT NONE
        INTEGER marknum

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER symnum, colornum, errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSPMI: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (marknum .LT. 1) .OR.
     .       (marknum .GT. maxmarkers) ) THEN
            errstr = 'FGD_GSPMI: invalid marknum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        symnum = markersymbol(marknum, activewindow) 
        IF ( (symnum .LT. 1) .OR.
     .       (symnum .GT. maxsymbolobjs) ) THEN
            errstr = 'FGD_GSPMI: invalid markersymbol'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( symbolobjs(symnum, activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPMI: null marker symbolobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        colornum = markercolor(marknum, activewindow)
        IF ( (colornum .LT. 1) .OR.
     .       (colornum .GT. maxcolorobjs) ) THEN
            errstr = 'FGD_GSPMI: invalid markercolor'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( colorobjs(colornum, activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPMI: null marker colorobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( markersize(marknum, activewindow) .LE. 0.0 ) THEN
            errstr = 'FGD_GSPMI: invalid markersize'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        activemarker = marknum

        RETURN
        END

