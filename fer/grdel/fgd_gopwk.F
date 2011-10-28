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

        CHARACTER*2048 errmsg
        INTEGER TM_LENSTR, engnamelen, wintitlelen,
     .                     showit, errmsglen
        REAL newwidth, newheight

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
        newwidth = windowwidth(windowid) * 1000.0
        newheight = windowheight(windowid) * 1000.0
        IF ( showwindow(windowid) ) THEN
            showit = 1
        ELSE
            showit = 0
        ENDIF
        CALL FGDWINCREATE(windowobjs(windowid),
     .                    enginename, engnamelen,
     .                    windowtitle, wintitlelen,
     .                    newwidth, newheight, showit)
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

        RETURN
        END

