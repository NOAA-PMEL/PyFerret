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

        REAL*4 dpix, dpiy
        INTEGER screenwidth, screenheight, success, errmsglen
        CHARACTER*2048 errmsg

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GQDSP: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'FGD_GQDSP: null windowobj'
        ENDIF

        CALL FGDWINSCREENINFO(success, windowobjs(windowid), 
     .                        dpix, dpiy, screenwidth, screenheight)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
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
