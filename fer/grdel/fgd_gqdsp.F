*
* Definition of GQDSP
* (inqure worstation display space dize)
*

        SUBROUTINE FGD_GQDSP(wstype, istat, dspunit,
     .                   dmaxx, dmaxy, imaxx, imaxy)

        IMPLICIT NONE
        INTEGER wstype, istat, dspunit, imaxx, imaxy
        REAL dmaxx, dmaxy

*       Canned values from XWindows display
        dspunit = 1
        dmaxx = 1280.0
        dmaxy = 1024.0
        imaxx = 1280
        imaxy = 1024
        istat = 0

        RETURN
        END
