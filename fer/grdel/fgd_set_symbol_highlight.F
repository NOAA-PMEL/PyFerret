*
* Assigns the color used to highlight (outline) symbols in the active window
*
* Input Arguments:
*     colorid - symbol highlight color ID; can be negative for no highlight
*
        SUBROUTINE FGD_SET_SYMBOL_HIGHLIGHT(colorid)

        IMPLICIT NONE
        INTEGER colorid

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, errstrlen, TM_LENSTR
        CHARACTER*256 errstr

        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_SYMBOL_HIGHLIGHT: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_SET_SYMBOL_HIGHLIGHT: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( colorid .GE. 0 ) THEN
            colornum = colorid + 1
            IF ( colornum .GT. maxcolorobjs ) THEN
                errstr = 'FGD_SET_SYMBOL_HIGHLIGHT: invalid colorid'
                errstrlen = TM_LENSTR(errstr)
                CALL SPLIT_LIST(pttmode_help,err_lun,errstr,errstrlen)
                RETURN
            ENDIF
            IF ( colorobjs(colornum, activewindow) .EQ. nullobj ) THEN
                errstr = 'FGD_SET_SYMBOL_HIGHLIGHT: invalid color'
                errstrlen = TM_LENSTR(errstr)
                CALL SPLIT_LIST(pttmode_help,err_lun,errstr,errstrlen)
                RETURN
            ENDIF
        ELSE
            colornum = 0
        ENDIF

*       Record the symbol highlight color number
        activesymhighlight = colornum

        RETURN
        END

