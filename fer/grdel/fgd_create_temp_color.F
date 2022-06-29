*
* Create a 2nd or 3rd color object for the given window
*
* 4/22 *acm*  PyFerret issue #51: Allow for defining up to 4 more colors/pens to use in 
*                    graticule lines; vertical and horizontal, and possibly different styles
*                    for large and small tics. Custom colors/lines have been limited to 1, 
*                    so custom-color line plots with custom-color grats were not implemented.
*                  extrapen is 0 (for original pen) 1, 2, 3, 4 (graticule pens)

        SUBROUTINE FGD_CREATE_TEMP_COLOR(windowid, icolor, 
     .                 redfrac, greenfrac, bluefrac, opaquefrac)

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER windowid, icolor, errstrlen, TM_LENSTR
        REAL*4 redfrac, greenfrac, bluefrac, opaquefrac
        CHARACTER*256 errstr

*       Sanity checks
        icolor = -1
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (redfrac .LT. 0.0) .OR.  (redfrac .GT. 1.0) ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: Invalid redfrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (greenfrac .LT. 0.0) .OR.  (greenfrac .GT. 1.0) ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: Invalid greenfrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (bluefrac .LT. 0.0) .OR.  (bluefrac .GT. 1.0) ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: Invalid bluefrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (opaquefrac .LT. 0.0) .OR.  (opaquefrac .GT. 1.0) ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: Invalid opaquefrac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

	! for now?
        IF ( extrapen .GT. 4 ) THEN
            errstr = 'FGD_CREATE_TEMP_COLOR: may only create 4 additional colors...'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

        icolor = maxcolorobjs - extrapen - 1
        CALL FGD_GSCR(windowid, icolor,
     .                redfrac, greenfrac, bluefrac, opaquefrac)

        RETURN
        END

