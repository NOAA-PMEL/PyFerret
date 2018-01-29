*
* Create a symbol object.
* (set polymarker representation)
*
* The marker size in GSK was a factor, so not very clear
* what the units should be.  The marker size in PyFerret
* are in units of points (1/72 inches); possibly further 
* scaled by the line width scaling factor.
*
* The symbol type character changed from a integer value 
* to a string (name for the symbol) to allow custom types.
*

        SUBROUTINE FGD_GSPMR(windowid, symnum, 
     .                       symname, symsize, colorindex)

        IMPLICIT NONE
        INTEGER windowid, symnum, colorindex
        REAL*4 symsize
        CHARACTER*(*) symname

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER TM_LENSTR

        INTEGER colornum, snlen, success, errmsglen
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

        snlen = TM_LENSTR(symname)
        IF ( snlen .LE. 0 ) THEN
            errmsg = 'No symbol name given'
            errmsglen = 20
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
            RETURN
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
        symbolsize(symnum, windowid) = symsize

        RETURN
        END
