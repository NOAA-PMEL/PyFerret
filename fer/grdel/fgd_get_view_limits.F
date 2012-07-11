*
* Return the limits of the active transform of the 
* active window in terms of both the view fractions
* and the user coordinates.
*

        SUBROUTINE FGD_GET_VIEW_LIMITS(
     .                 lftfrc, rgtfrc, btmfrc, topfrc,
     .                 lftcrd, rgtcrd, btmcrd, topcrd)

        IMPLICIT NONE

        REAL*4 lftfrc, rgtfrc, btmfrc, topfrc,
     .         lftcrd, rgtcrd, btmcrd, topcrd

        INCLUDE 'fgrdel.cmn'

*       Sanity checks - transform 0 is an implicit transform
        IF ( (activetransform .LT. 0) .OR.
     .       (activetransform .GT. maxtransforms) ) THEN
            STOP 'FGD_GET_VIEW_LIMTS: No active transform'
        ENDIF
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GET_VIEW_LIMITS: No active window'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GET_VIEW_LIMITS: null active windowobj'
        ENDIF

        IF ( activetransform .NE. 0 ) THEN
*           Get the view limits and adjust, realizing that the
*           full window is thought of as going from [0.0,0.0]
*           to [ndcxmax,ndcymax] (set by FGD_GSWKWN).
            lftfrc = viewleftfrac(activetransform) / 
     .                ndcxmax(activewindow)
            rgtfrc = viewrightfrac(activetransform) / 
     .                ndcxmax(activewindow)
            btmfrc = viewbottomfrac(activetransform) / 
     .                ndcymax(activewindow)
            topfrc = viewtopfrac(activetransform) / 
     .                ndcymax(activewindow)
            lftcrd = viewleftcoord(activetransform)
            btmcrd = viewbottomcoord(activetransform)
            rgtcrd = viewrightcoord(activetransform)
            topcrd = viewtopcoord(activetransform)
            IF ( (lftfrc .LT. 0.0) .OR.
     .           (lftfrc .GE. rgtfrc) .OR.
     .           (rgtfrc .GT. 1.0) .OR.
     .           (btmfrc .LT. 0.0) .OR.
     .           (btmfrc .GE. topfrc) .OR.
     .           (topfrc .GT. 1.0) ) THEN
                WRITE(*,*) 'lft, btm, rgt, top fracs = ',
     .                     lftfrc, btmfrc, rgtfrc, topfrc
                WRITE(*,*) 'lft, btm, rgt, top coords = ',
     .                     lftcrd, btmcrd, rgtcrd, topcrd
                STOP 'FGD_GET_VIEW_LIMITS: invalid adj view fracs'
            ENDIF
        ELSE
*           The identity transform.
            lftfrc = 0.0
            btmfrc = 0.0
            rgtfrc = 1.0
            topfrc = 1.0
            lftcrd = 0.0
            btmcrd = 0.0
            rgtcrd = windowwidth(activewindow) *
     .                 windowdpix(activewindow)
            topcrd = windowheight(activewindow) *
     .                 windowdpiy(activewindow)
        ENDIF

        RETURN
        END

