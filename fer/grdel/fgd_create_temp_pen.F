*
* Create a temporary pen and symbol object (dot) 
* for the default window using the given color
*
* 4/22 *acm*  PyFerret issue #51: Allow for defining up to 4 more colors/pens to use in 
*                    graticule lines; vertical and horizontal, and possibly different styles
*                    for large and small tics. Custom colors/lines have been limited to 1, 
*                    so custom-color line plots with custom-color grats were not implemented.
*                  extrapen is 0 (for original pen) 1, 2, 3, 4 (graticule pens)

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

        ipen = maxpenobjs - extrapen

*       linestyle = 1 is a solid line
        CALL FGD_GSPLR(windowid, ipen, 1, thickness, icolor)

        RETURN
        END

