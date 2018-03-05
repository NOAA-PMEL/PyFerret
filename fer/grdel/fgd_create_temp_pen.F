*
* Create a temporary pen and symbol object (dot) 
* for the default window using the given color
*

        SUBROUTINE FGD_CREATE_TEMP_PEN(windowid, 
     .                        ipen, icolor, thickness)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER windowid, ipen, icolor, errstrlen, TM_LENSTR
        REAL*4 thickness
        REAL*4 dotsize
        CHARACTER*256 errstr

*       Sanity checks
        ipen = -1
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_CREATE_TEMP_PEN: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_CREATE_TEMP_PEN: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (icolor .LT. 0) .OR.
     .       (icolor .GE. maxcolorobjs) ) THEN
            errstr = 'FGD_CREATE_TEMP_PEN: invalid icolor'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( colorobjs(icolor+1, windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_CREATE_TEMP_PEN: null colorobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        ipen = maxpenobjs
*       linestyle = 1 is a solid line
        CALL FGD_GSPLR(windowid, ipen, 1, thickness, icolor)

        RETURN
        END

