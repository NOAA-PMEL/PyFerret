*
* Definition of GQDSP
* (inqure worstation display space size)
* Returns the DPI and size of the default screen (display) 
* for the given window.
*    dmaxx and dmaxy are sizes in meters
*    imaxx and imaxy are sizes in pixels
*

        SUBROUTINE FGD_GQDSP(windowid, istat, dspunit,
     .                   dmaxx, dmaxy, imaxx, imaxy)

        IMPLICIT NONE

        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER windowid, istat, dspunit, imaxx, imaxy
        REAL*4 dmaxx, dmaxy

        INTEGER TM_LENSTR

        REAL*4 dpix, dpiy
        INTEGER screenwidth, screenheight, success, errstrlen
        CHARACTER*2048 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'FGD_GQDSP: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'FGD_GQDSP: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF

        CALL FGDWINSCREENINFO(success, windowobjs(windowid), 
     .                        dpix, dpiy, screenwidth, screenheight)
        IF ( success .EQ. 0 ) THEN
            errstr = ' '
            CALL FGDERRMSG(errstr, errstrlen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            istat = 1
            RETURN
        ENDIF

*       Save the dpix and dpiy for use elsewhere 
*       so this routine does need to be called all the time
        windowdpix(windowid) = dpix
        windowdpiy(windowid) = dpiy

        dspunit = 1
        dmaxx = (screenwidth * 0.0254) / dpix
        dmaxy = (screenheight * 0.0254) / dpiy
        imaxx = screenwidth
        imaxy = screenheight
        istat = 0

        RETURN
        END
