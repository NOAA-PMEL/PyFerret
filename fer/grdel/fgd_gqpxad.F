*
* Inquire pixel array dimensions from user (world) coordinates.
*

        SUBROUTINE FGD_GQPXAD(windowid, left, top, right, bottom,
     .                        istat, widthpix, heightpix)

        IMPLICIT NONE
        INTEGER windowid, istat, widthpix, heightpix
        REAL*4 left, top, right, bottom

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        REAL*4 mywidth, myheight
        INTEGER success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GQPXAD: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GQPXAD: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
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
