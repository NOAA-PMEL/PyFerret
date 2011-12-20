*
* Select current (normalization) transformation number
*

        SUBROUTINE FGD_GSELNT(transnum)

        IMPLICIT NONE
        INTEGER transnum

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL lftfrac, rgtfrac, btmfrac, topfrac,
     .       lftcoord, rgtcoord, btmcoord, topcoord
        INTEGER clipit, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks - transform 0 is an implicit identity transform
        IF ( (transnum .LT. 0) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            STOP 'FGD_GSELNT: Invalid transnum value'
        ENDIF
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSELNT: No active window'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GSELNT: null active windowobj'
        ENDIF

*       Set this transform number to be the active one
        activetransform = transnum

*       Get the view limits of this transform
        CALL FGD_GET_VIEW_LIMITS(lftfrac, rgtfrac, btmfrac, topfrac,
     .                       lftcoord, rgtcoord, btmcoord, topcoord)

*       End any view that may currently be active
*       There might not be one started, so ignore any errors
        CALL FGDVIEWEND(success, windowobjs(activewindow))

*       Begin the new view
        IF ( cliptoview(activewindow) ) THEN
            clipit = 1
        ELSE
            clipit = 0
        ENDIF
        CALL FGDVIEWBEGIN(success, windowobjs(activewindow),
     .                    lftfrac, btmfrac, rgtfrac, topfrac,
     .                    clipit)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        somethingdrawn = .FALSE.

        RETURN
        END

