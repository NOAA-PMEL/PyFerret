*
* Creates a Window object for a given graphics engine.
* (Initializes the graphics engine if needed.)
*
* Input arguments:
*     windowid: ID number for this window (1, 2, ...)
*     connectid: ignored
*     windowtype: ignored
*
* Values used from fgrdel.cmn:
*     enginename
*     windowtitle
*     windowwidth
*     windowheight
*     showwindow
*

        SUBROUTINE FGD_GOPWK(windowid, connectid, windowtype)

        IMPLICIT NONE
        INTEGER windowid, connectid, windowtype

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER TM_LENSTR

        INTEGER engnamelen, wintitlelen, showit,
     .          success, errmsglen, aaint
        CHARACTER*2048 errmsg
        REAL wdth, hght, dpix, dpiy

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GOPWK: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
            STOP 'FGD_GOPWK: non-null windowobj'
        ENDIF

*       Create the window
        engnamelen = TM_LENSTR(enginename)
        wintitlelen = TM_LENSTR(windowtitle)
        IF ( showwindow(windowid) ) THEN
            showit = 1
        ELSE
            showit = 0
        ENDIF
        CALL FGDWINCREATE(windowobjs(windowid),
     .                    enginename, engnamelen,
     .                    windowtitle, wintitlelen, showit)
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            RETURN
        ENDIF

*       Create opaque white as the first color for this window
        CALL FGDCOLOR(colorobjs(1, windowid), windowobjs(windowid),
     .                1.0, 1.0, 1.0, 1.0)
        IF ( colorobjs(1, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
*       Create opaque black as the second color for this window
        CALL FGDCOLOR(colorobjs(2, windowid), windowobjs(windowid),
     .                0.0, 0.0, 0.0, 1.0)
        IF ( colorobjs(2, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
*       Get the horizontal and vertical DPI,
*       and store it so this is called only once.
        CALL FGDWINDPI(success, windowobjs(windowid), dpix, dpiy)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        windowdpix(windowid) = dpix
        windowdpiy(windowid) = dpiy
*       Resize the window based on these DPI values
        wdth = windowwidth(windowid) * dpix
        hght = windowheight(windowid) * dpiy
        CALL FGDWINSETSIZE(success, windowobjs(windowid), wdth, hght)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
*       Set the default antialias state
        IF ( useantialiasing ) THEN
            aaint = 1
        ELSE
            aaint = 0
        ENDIF
        CALL FGDWINSETANTIALIAS(success, windowobjs(windowid), aaint)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF
        RETURN
        END

