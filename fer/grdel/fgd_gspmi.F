*
* Set the active sybmol for markers in the active window
* (set polymarker index)
*

        SUBROUTINE FGD_GSPMI(symnum)

        IMPLICIT NONE
        INTEGER symnum

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSPMI: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (symnum .LT. 1) .OR.
     .       (symnum .GT. maxsymbolobjs) ) THEN
            errstr = 'FGD_GSPMI: invalid symbol number'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( symbolobjs(symnum, activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSPMI: invalid symbol'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        activesymbol = symnum

        RETURN
        END

