*
* Returns the RGBA fractions of the color object in a pen object 
* with ID number pennum for the window with ID number windowid.  
* If the pen object exists the RGBA fractions are returned 
* in redfrac, greenfrac, bluefrac, and opaquefrac arguments, 
* and istat is set to zero.  If the pen object does not exist, 
* then istat is set to one and the RGBA fractions are unchanged.
*

        SUBROUTINE FGD_GQPEN(windowid, pennum, istat,
     .                       redfrac, greenfrac, bluefrac, opaquefrac)

        IMPLICIT NONE
        INTEGER windowid, pennum, istat
        REAL*4 redfrac, greenfrac, bluefrac, opaquefrac

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER colornum, errstrlen, TM_LENSTR
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GQPEN: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GQPEN: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF
        IF ( (pennum .LE. 0) .OR.
     .       (pennum .GT. maxpenobjs) ) THEN
            errstr = 'FGD_GQPEN: invalid pennum'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF
        IF ( penobjs(pennum, windowid) .EQ. nullobj ) THEN
            istat = 1
            RETURN
        ENDIF

*       Simplest approach for retrieving pen color numbers
*       and the color RGBA values for that color was just 
*       to store them in the common block when they are set
        colornum = pencolor(pennum, windowid)
        redfrac = colorrgba(1, colornum, windowid)
        greenfrac = colorrgba(2, colornum, windowid)
        bluefrac = colorrgba(3, colornum, windowid)
        opaquefrac = colorrgba(4, colornum, windowid)
        istat = 0

        RETURN
        END

