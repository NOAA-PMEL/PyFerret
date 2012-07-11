*
* Create a symbol object.
* (set polymarker representation)
*

        SUBROUTINE FGD_GSPMR(windowid, symnum, 
     .                 symtype, symsize, colorindex)

        IMPLICIT NONE
        INTEGER windowid, symnum, symtype, colorindex
        REAL symsize

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, snlen, success, errmsglen
        CHARACTER*8 symname
        REAL adjsize
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSPMR: Invalid windowid'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSPMR: null windowobj'
        ENDIF
        IF ( (symnum .LT. 1) .OR.
     .       (symnum .GT. maxsymbolobjs) ) THEN
            STOP 'FGD_GSPMR: Invalid symnum'
        ENDIF
        IF ( (colorindex .LT. 0) .OR.
     .       (colorindex .GE. maxcolorobjs) ) THEN
            STOP 'FGD_GSPMR: Invalid colorindex'
        ENDIF

        colornum = colorindex + 1
        IF ( colorobjs(colornum, windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GSPMR: null colorobj'
        ENDIF

*       symbol types
        IF ( symtype .EQ. 1 ) THEN
*           small filled circle
            symname = '.'
            snlen = 1
        ELSE IF ( symtype .EQ. 2 ) THEN
*           plus
            symname = '+'
            snlen = 1
        ELSE IF ( symtype .EQ. 3 ) THEN
*           asterisk
            symname = '*'
            snlen = 1
        ELSE IF ( symtype .EQ. 4 ) THEN
*           open circle
            symname = 'o'
            snlen = 1
        ELSE IF ( symtype .EQ. 5 ) THEN
*           ex
            symname = 'x'
            snlen = 1
        ELSE IF ( symtype .EQ. 6 ) THEN
*           triangle
            symname = '^'
            snlen = 1
        ELSE IF ( symtype .EQ. 7 ) THEN
*           square
            symname = '#'
            snlen = 1
        ELSE
            STOP 'FGD_GSPMR: invalid symtype'
        ENDIF

*       Adjust the symbol size to typical user coordinates
        adjsize = 10.0 * symsize

*       If a symbol object already exists, delete it
        IF ( symbolobjs(symnum, windowid) .NE. nullobj ) THEN
            CALL FGDSYMBOLDEL(success, symbolobjs(symnum, windowid))
            IF ( success .EQ. 0 ) THEN
                errmsg = ' '
                CALL FGDERRMSG(errmsg, errmsglen)
                CALL SPLIT_LIST(pttmode_help, err_lun,
     .                          errmsg, errmsglen)
            ENDIF
            symbolobjs(symnum, windowid) = nullobj
        ENDIF

*       Create the symbol object
        CALL FGDSYMBOL(symbolobjs(symnum, windowid),
     .                 windowobjs(windowid), symname, snlen)
        IF ( symbolobjs(symnum, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        symbolcolor(symnum, windowid) = colornum
        symbolsize(symnum, windowid) = adjsize

        RETURN
        END
