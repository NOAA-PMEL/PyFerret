*
* Get the PyFerret symbol number that matches the given
* symbol name for the currently active window.
* Creates the symbol object if it does not already exist.
*
* Input Arguments:
*     symname: name of the symbol to make active
*     symnamelen: length of the symbol name
* Returns: 
*     if successful, the matching symbol number;
*     otherwise zero, and an error message was printed
*
        INTEGER FUNCTION FGD_GET_SYMBOLNUM(symname, symnamelen)

        IMPLICIT NONE
        CHARACTER*(*) symname
        INTEGER symnamelen

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'xplot_state.cmn'
        INCLUDE 'fgrdel.cmn'

*       function declaration
        INTEGER FGDSYMBOLMATCHES, TM_LENSTR

*       local variables
        INTEGER errstrlen, k;
        CHARACTER errstr*2048;

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GET_SYMBOLNUM: invalid active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            FGD_GET_SYMBOLNUM = 0
            RETURN
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            errstr = 'FGD_GET_SYMBOLNUM: no active window'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            FGD_GET_SYMBOLNUM = 0
            RETURN
        ENDIF

*       Check if the symbol object already exists
        DO k = 1, maxsymbolobjs
            IF ( symbolobjs(k, activewindow) .NE. nullobj ) THEN
                IF ( FGDSYMBOLMATCHES(symbolobjs(k, activewindow), 
     .                        symname, symnamelen) .NE. 0 ) THEN
*                   Symbol found
                    FGD_GET_SYMBOLNUM = k
                    RETURN
                ENDIF
            ENDIF
        ENDDO

*       No match found; try to create        
        DO k = 1, maxsymbolobjs
            IF ( symbolobjs(k, activewindow) .EQ. nullobj ) THEN
                CALL FGDSYMBOL(symbolobjs(k, activewindow), 
     .                  windowobjs(activewindow), symname, symnamelen)
                IF ( symbolobjs(k, activewindow) .EQ. nullobj ) THEN
*                   Symbol cannot be created
                    errstr = ' '
                    CALL FGDERRMSG(errstr, errstrlen)
                    CALL SPLIT_LIST(pttmode_help, err_lun, 
     .                              errstr, errstrlen)
                    FGD_GET_SYMBOLNUM = 0
                    RETURN
                ENDIF
*               Symbol created
                FGD_GET_SYMBOLNUM = k
                RETURN
            ENDIF
        ENDDO

*       No space available for another symbol
        errstr = 'FGD_GET_SYMBOLNUM: no space for another symbol'
        errstrlen = TM_LENSTR(errstr)
        CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
        FGD_GET_SYMBOLNUM = 0
        RETURN

        END

