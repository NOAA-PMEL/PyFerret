*
* Inquire pixel array dimensions from user (world) coordinates.
*

        SUBROUTINE FGD_GQPXAD(windowid, left, top, right, bottom,
     .                        istat, widthpix, heightpix)

        IMPLICIT NONE
        INTEGER windowid, istat, widthpix, heightpix
        REAL left, top, right, bottom

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL mywidth, myheight
        INTEGER success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GQPXAD: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GQPXAD: null windowobj'
        ENDIF

        mywidth = right - left
        myheight = top - bottom
        IF ( activetransform .GT. 0 ) THEN
            mywidth = mywidth * ( viewrightfrac(activetransform)
     .                           - viewleftfrac(activetransform) )
            mywidth = mywidth / ( viewrightcoord(activetransform)
     .                           - viewleftcoord(activetransform) )
            mywidth = mywidth / ndcxmax(windowid)
            myheight = myheight * ( viewtopfrac(activetransform)
     .                            - viewbottomfrac(activetransform) )
            myheight = myheight / ( viewtopcoord(activetransform)
     .                            - viewbottomcoord(activetransform) )
            myheight = myheight / ndcymax(windowid)
        ENDIF
        mywidth = mywidth * windowwidth(windowid) * 
     .                      windowdpix(windowid)
        myheight = myheight * windowheight(windowid) * 
     .                        windowdpiy(windowid)

*       make sure the width and height are positive integers
        widthpix = INT(ABS(mywidth))
        heightpix = INT(ABS(myheight))
        istat = 0
        RETURN
        END
