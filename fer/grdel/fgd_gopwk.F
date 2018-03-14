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
*     rasteronly
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

        INTEGER engnamelen, wintitlelen, showit, noalpha, 
     .          success, errstrlen, aaint, roint, namelen
        CHARACTER*2048 errstr
        CHARACTER*16 symname
        REAL*4 redf, greenf, bluef, opaquef

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GOPWK: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .NE. nullobj ) THEN
            errstr = 'FGD_GOPWK: non-null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Create the window
        engnamelen = TM_LENSTR(enginename(windowid))
        wintitlelen = TM_LENSTR(windowtitle)
        showit = 1
*       ignoring alpha in colors ?
        noalpha = 0
*       raster only ?
        IF ( rasteronly ) THEN
            roint = 1
        ELSE
            roint = 0
        ENDIF
        CALL FGDWINCREATE(windowobjs(windowid),
     .                    enginename(windowid), engnamelen,
     .                    windowtitle, wintitlelen, showit, 
     .                    noalpha, roint)
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
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
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        colorrgba(1, 1, windowid) = redf
        colorrgba(2, 1, windowid) = greenf
        colorrgba(3, 1, windowid) = bluef
        colorrgba(4, 1, windowid) = opaquef

*       Create opaque black as the second color for this window
        redf = 0.0
        greenf = 0.0
        bluef = 0.0
        opaquef = 1.0
        CALL FGDCOLOR(colorobjs(2, windowid), windowobjs(windowid),
     .                redf, greenf, bluef, opaquef)
        IF ( colorobjs(2, windowid) .EQ. nullobj ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        colorrgba(1, 2, windowid) = redf
        colorrgba(2, 2, windowid) = greenf
        colorrgba(3, 2, windowid) = bluef
        colorrgba(4, 2, windowid) = opaquef

*       Set the antialias state
        IF ( antialias(windowid) ) THEN
            aaint = 1
        ELSE
            aaint = 0
        ENDIF
        CALL FGDWINSETANTIALIAS(success, windowobjs(windowid), aaint)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

*       Set the scaling factor for line widths and symbol sizes
        CALL FGDWINSETWIDTHFACTOR(success, windowobjs(windowid), 
     .                            thickfactor(windowid))
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

*       Create named symbol objects for this window.  The named symbol 
*       "dot" must be defined and must be the first symbol for each 
*       window, so give an error message there was a problem creating it.
        symname = "dot"
        namelen = 3
        CALL FGDSYMBOL(symbolobjs(1, windowid), 
     .                 windowobjs(windowid), symname, namelen)
        IF ( symbolobjs(1, windowid) .EQ. nullobj ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
*       All other symbols are optional, so do not give error messages
*       if there was a problem creating them at this point.  The 
*       symbols "dotex", "dotplus", "circle", "circfill", "circex", 
*       and "circplus" are pre-defined in the engines currently used.
        symname = "dotex"
        namelen = 5
        CALL FGDSYMBOL(symbolobjs(2, windowid), 
     .                 windowobjs(windowid), symname, namelen)
        symname = "dotplus"
        namelen = 7
        CALL FGDSYMBOL(symbolobjs(3, windowid), 
     .                 windowobjs(windowid), symname, namelen)
        symname = "circle"
        namelen = 6
        CALL FGDSYMBOL(symbolobjs(4, windowid), 
     .                 windowobjs(windowid), symname, namelen)
        symname = "circfill"
        namelen = 8
        CALL FGDSYMBOL(symbolobjs(5, windowid), 
     .                 windowobjs(windowid), symname, namelen)
        symname = "circex"
        namelen = 6
        CALL FGDSYMBOL(symbolobjs(6, windowid), 
     .                 windowobjs(windowid), symname, namelen)
        symname = "circplus"
        namelen = 8
        CALL FGDSYMBOL(symbolobjs(7, windowid), 
     .                 windowobjs(windowid), symname, namelen)
*       Symbols from definition files will be added when first used.

*       Set the background color
        CALL FGDWINCLEAR(success, windowobjs(windowid), 
     .                   colorobjs(1,windowid))
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF

        RETURN
        END

