*
* Select current (normalization) transformation number
*

        SUBROUTINE FGD_GSELNT(transnum)

        IMPLICIT NONE
        INTEGER transnum

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL*4 lftfrac, rgtfrac, btmfrac, topfrac,
     .         lftcoord, rgtcoord, btmcoord, topcoord
        INTEGER clipit, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks - transform 0 is an implicit identity transform
        IF ( (transnum .LT. 0) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            errstr = 'FGD_GSELNT: invalid transnum value'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GSELNT: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GSELNT: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
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
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        somethingdrawn = .FALSE.

        RETURN
        END

