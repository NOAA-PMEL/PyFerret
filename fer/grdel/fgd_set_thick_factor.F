*
* Assigns the scaling factor for line widths and symbol sizes
* in the specified window
*
* Input Arguments:
*     windowid - number of the window to use
*     widthfactor - scaling factor to use
*
        SUBROUTINE FGD_SET_THICK_FACTOR(windowid, widthfactor)

        IMPLICIT NONE
        INTEGER windowid
        REAL*4 widthfactor

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       Local variables
        INTEGER success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_SET_THICK_FACTOR: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( widthfactor .LE. 0.0 ) THEN
            errstr = 'FGD_SET_THICK_FACTOR: Invalid widthfactor value'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       This may be called to prior to window creation
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
*           Update the width scaling factor of this window
            CALL FGDWINSETWIDTHFACTOR(success, 
     .                    windowobjs(windowid), widthfactor)
            IF ( success .EQ. 0 ) THEN
*               Error - output error message
                errstr = ' '
                CALL FGDERRMSG(errstr, errstrlen)
                CALL SPLIT_LIST(pttmode_help, 
     .                          err_lun, errstr, errstrlen)
                RETURN
            ENDIF
        ENDIF

*       Record the width scaling factor for this window
        thickfactor(windowid) = widthfactor

        RETURN
        END

