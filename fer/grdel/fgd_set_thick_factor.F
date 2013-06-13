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
        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SET_THICK_FACTOR: Invalid windowid value'
        ENDIF
        IF ( widthfactor .LE. 0.0 ) THEN
            STOP 'FGD_SET_THICK_FACTOR: Invalid widthfactor value'
        ENDIF

*       This may be called to prior to window creation
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
*           Update the width scaling factor of this window
            CALL FGDWINSETWIDTHFACTOR(success, 
     .                    windowobjs(windowid), widthfactor)
            IF ( success .EQ. 0 ) THEN
*               Error - output error message
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, 
     .                          err_lun, errmsg, errmsglen)
            ENDIF
        ENDIF

*       Record the width scaling factor for this window
        thickfactor(windowid) = widthfactor

        RETURN
        END

