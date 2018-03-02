*
* Draw a polyline in the active window using the active pen object
*

        SUBROUTINE FGD_GPL(npts, ptsx, ptsy)

        IMPLICIT NONE
        INTEGER npts
        REAL*4 ptsx, ptsy
        DIMENSION ptsx(npts), ptsy(npts)

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER k, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GPL: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GPL: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        IF ( (activepen .LT. 1) .OR.
     .       (activepen .GT. maxpenobjs) ) THEN
            errstr = 'FGD_GPL: no activepen'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( penobjs(activepen, activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GPL: invalid activepen'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Draw polyline
        CALL FGDDRAWMULTILINE(success, windowobjs(activewindow),
     .          ptsx, ptsy, npts, penobjs(activepen, activewindow))
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        somethingdrawn = .TRUE.

        RETURN
        END

