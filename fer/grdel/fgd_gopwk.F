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
*     antialias
*     thickfactor
*

        SUBROUTINE FGD_GOPWK(windowid, connectid, windowtype)

        IMPLICIT NONE
        INTEGER windowid, connectid, windowtype

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER TM_LENSTR

        INTEGER engnamelen, wintitlelen, showit, 
     .          noalpha, success, errmsglen, aaint
        CHARACTER*2048 errmsg
        REAL*4 redf, greenf, bluef, opaquef

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GOPWK: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
            STOP 'FGD_GOPWK: non-null windowobj'
        ENDIF

*       Create the window
        engnamelen = TM_LENSTR(enginename(windowid))
        wintitlelen = TM_LENSTR(windowtitle)
        showit = 1
*       ignoring alpha in colors ?
        noalpha = 0
        CALL FGDWINCREATE(windowobjs(windowid),
     .                    enginename(windowid), engnamelen,
     .                    windowtitle, wintitlelen, showit, noalpha)
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            RETURN
        ENDIF

*       Create opaque white as the first color for this window
        redf = 1.0
        greenf = 1.0
        bluef = 1.0
        opaquef = 1.0
        CALL FGDCOLOR(colorobjs(1, windowid), windowobjs(windowid),
     .                redf, greenf, bluef, opaquef)
        IF ( colorobjs(1, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

*       Create opaque black as the second color for this window
        redf = 0.0
        greenf = 0.0
        bluef = 0.0
        opaquef = 1.0
        CALL FGDCOLOR(colorobjs(2, windowid), windowobjs(windowid),
     .                redf, greenf, bluef, opaquef)
        IF ( colorobjs(2, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

*       Set the antialias state
        IF ( antialias(windowid) ) THEN
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

*       Set the scaling factor for line widths and symbol sizes
        CALL FGDWINSETWIDTHFACTOR(success, windowobjs(windowid), 
     .                            thickfactor(windowid))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

*       Set the background color
        CALL FGDWINCLEAR(success, windowobjs(windowid), 
     .                   colorobjs(1,windowid))
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END

