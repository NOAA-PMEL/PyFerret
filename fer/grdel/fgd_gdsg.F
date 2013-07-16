*
* Definition of GDSG (delete segment).
*
        SUBROUTINE FGD_GDSG(segnum)

        IMPLICIT NONE
        INTEGER segnum

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GCRSQ: Invalid activewindow value'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GCRSQ: null windowobj'
        ENDIF

        CALL FGDSEGDELETE(success, windowobjs(activewindow), segnum)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF


        RETURN
        END

