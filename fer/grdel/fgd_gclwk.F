*
* Deletes (closes and destroys) a Window created by FGD_GOPWK
* Also deletes any color, brush, font, pen, and symbol objects
* associated with this window
*
* Input Arguments:
*     windowid: ID number of the Window to be closed
*
        SUBROUTINE FGD_GCLWK(windowid)

        IMPLICIT NONE
        INTEGER windowid

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER k, success, errstrlen, TM_LENSTR
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GCLWK: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GCLWK: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Delete any remaining objects associated with this window.
*       Clear values associated with those objects just to be safe.
*       Error messages are printed just as warnings that something
*       is not right, but errors are otherwise ignored.
        DO 10 k = 1, maxsymbolobjs
            IF ( symbolobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDSYMBOLDEL(success, symbolobjs(k, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun, 
     .                              errstr, errstrlen)
                ENDIF
                symbolobjs(k, windowid) = nullobj
            ENDIF
   10   CONTINUE
        DO 20 k = 1, maxpenobjs
            IF ( penobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDPENDEL(success, penobjs(k, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun, 
     .                              errstr, errstrlen)
                ENDIF
                penobjs(k, windowid) = nullobj
            ENDIF
            pencolor(k, windowid) = 0
            penwidth(k, windowid) = -1.0
   20   CONTINUE
        DO 30 k = 1, maxfontobjs
            IF ( fontobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDFONTDEL(success, fontobjs(k, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun, 
     .                              errstr, errstrlen)
                ENDIF
                fontobjs(k, windowid) = nullobj
            ENDIF
   30   CONTINUE
        DO 40 k = 1, maxbrushobjs
            IF ( brushobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDBRUSHDEL(success, brushobjs(k, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun, 
     .                              errstr, errstrlen)
                ENDIF
                brushobjs(k, windowid) = nullobj
            ENDIF
            brushcolor(k, windowid) = 0
   40   CONTINUE
        DO 50 k = 1, maxcolorobjs
            IF ( colorobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDCOLORDEL(success, colorobjs(k, windowid))
                IF ( success .EQ. 0 ) THEN
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun, 
     .                              errstr, errstrlen)
                ENDIF
                colorobjs(k, windowid) = nullobj
            ENDIF
            colorrgba(1, k, windowid) = -1.0
            colorrgba(2, k, windowid) = -1.0
            colorrgba(3, k, windowid) = -1.0
            colorrgba(4, k, windowid) = -1.0
   50   CONTINUE

*       Clear the DPI values just to be safe
        windowdpix(windowid) = 0.0
        windowdpiy(windowid) = 0.0

*       Delete the window
        CALL FGDWINDELETE(success, windowobjs(windowid))
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        ENDIF
        windowobjs(windowid) = nullobj

        RETURN
        END

