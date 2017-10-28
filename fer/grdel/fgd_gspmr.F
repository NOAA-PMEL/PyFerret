*
* Create a symbol object.
* (set polymarker representation)
*
* The marker size in GSK was a factor, so not very clear
* what the units should be.  The marker size in PyFerret
* are in units of points (1/72 inches); possibly further 
* scaled by the line width scaling factor.
*

        SUBROUTINE FGD_GSPMR(windowid, symnum, 
     .                 symtype, symsize, colorindex)

        IMPLICIT NONE
        INTEGER windowid, symnum, symtype, colorindex
        REAL*4 symsize

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, snlen, success, errmsglen
        CHARACTER*8 symname
        CHARACTER*2048 errmsg
*       For custom symbols: ptsx, ptsy, and numpts describes 
*       the symbol as a multiline drawing on a [0,100] square grid
        REAL*4 ptsx(32), ptsy(32)
        INTEGER numpts

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
            numpts = 0
            symname = '.'
            snlen = 1
        ELSE IF ( symtype .EQ. 2 ) THEN
*           plus
            numpts = 0
            symname = '+'
            snlen = 1
        ELSE IF ( symtype .EQ. 3 ) THEN
*           asterisk
            numpts = 0
            symname = '*'
            snlen = 1
        ELSE IF ( symtype .EQ. 4 ) THEN
*           open circle
            numpts = 0
            symname = 'o'
            snlen = 1
        ELSE IF ( symtype .EQ. 5 ) THEN
*           ex
            numpts = 0
            symname = 'x'
            snlen = 1
        ELSE IF ( symtype .EQ. 6 ) THEN
*           triangle
            numpts = 0
            symname = '^'
            snlen = 1
        ELSE IF ( symtype .EQ. 7 ) THEN
*           square
            numpts = 0
            symname = '#'
            snlen = 1
        ELSE
            STOP 'FGD_GSPMR: invalid symtype'
        ENDIF

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
     .                 windowobjs(windowid), symname, snlen,
     .                 ptsx, ptsy, numpts)
        IF ( symbolobjs(symnum, windowid) .EQ. nullobj ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        symbolcolor(symnum, windowid) = colornum
        symbolsize(symnum, windowid) = symsize

        RETURN
        END
