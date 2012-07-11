*
* Draw a polyline in the active window using the active pen object
*

        SUBROUTINE FGD_GPL(npts, ptsx, ptsy)

        IMPLICIT NONE
        INTEGER npts
        REAL ptsx, ptsy
        DIMENSION ptsx(npts), ptsy(npts)

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER k, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GPL: Invalid activewindow'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GPL: null activewindow'
        ENDIF

        IF ( (activepen .LT. 1) .OR.
     .       (activepen .GT. maxpenobjs) ) THEN
            STOP 'FGD_GPL: invalid activepen'
        ENDIF
        IF ( penobjs(activepen, activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GPL: null activepen'
        ENDIF

*       Draw polyline
        CALL FGDDRAWMULTILINE(success, windowobjs(activewindow),
     .          ptsx, ptsy, npts, penobjs(activepen, activewindow))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        somethingdrawn = .TRUE.

        RETURN
        END

