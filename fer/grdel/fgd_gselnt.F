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
     .       lftcoord, rgtcoord, btmcoord, topcoord,
     .       dpix, dpiy
        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks - transform 0 is an implicit transform
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

        activetransform = transnum
*       Get the view limits and 
*       adjust the view fractions by the limits defined by FGD_GSWKWN
        IF ( transnum .NE. 0 ) THEN
            lftfrac = viewleftfrac(transnum) / ndcxmax(activewindow)
            rgtfrac = viewrightfrac(transnum) / ndcxmax(activewindow)
            btmfrac = viewbottomfrac(transnum) / ndcymax(activewindow)
            topfrac = viewtopfrac(transnum) / ndcymax(activewindow)
            lftcoord = viewleftcoord(transnum)
            btmcoord = viewbottomcoord(transnum)
            rgtcoord = viewrightcoord(transnum)
            topcoord = viewtopcoord(transnum)
        ELSE
*           This should be the identity transform.
*           Normally this is only used to end a view.
            lftfrac = 0.0
            btmfrac = 0.0
            rgtfrac = 1.0
            topfrac = 1.0
            lftcoord = 0.0
            btmcoord = 0.0
            CALL FGD_GETDPI(activewindow, dpix, dpiy)
            rgtcoord = windowwidth(activewindow) * dpix *
     .                 ndcxmax(activewindow)
            topcoord = windowheight(activewindow) * dpiy *
     .                 ndcymax(activewindow)
        ENDIF
*       End any view that may currently be active
*       There might not be one started, so ignore any errors
        CALL FGDVIEWEND(success, windowobjs(activewindow))
*       Begin the new view
        CALL FGDVIEWBEGIN(success, windowobjs(activewindow),
     .                    lftfrac, btmfrac, rgtfrac, topfrac,
     .                    lftcoord, btmcoord, rgtcoord, topcoord)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        somethingdrawn = .FALSE.

        RETURN
        END

