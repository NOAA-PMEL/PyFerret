*
* Update a Window.
* Also ensures the window is visible
* (if appropriate for the engine).
*
* Input arguments:
*    windowid: window ID number to update
*    regenflag: ignored
*

        SUBROUTINE FGD_GUWK(windowid, regenflag)

        IMPLICIT NONE
        INTEGER windowid
        INTEGER regenflag

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER showit, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GUWK: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GUWK: null windowobj'
        ENDIF

*       Make the call to update the graphics
        CALL FGDWINUPDATE(success, windowobjs(windowid))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

*       Make the call to show the window (ignored if not appropriate)
        showit = 1
        CALL FGDWINSETVIS(success, windowobjs(windowid), showit)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

