*
* Definition of GINCH (initialize choices)
* Does nothing.
*
        SUBROUTINE FGD_GINCH(windowid, chdnr, istat, ichnr, pet,
     .                   xmin, xmax, ymin, ymax, ldr, datarec)

        IMPLICIT NONE
        INTEGER windowid, chdnr, istat, ichnr, pet, ldr
        REAL*4    xmin, xmax, ymin, ymax
        CHARACTER*(80) datarec
        DIMENSION datarec(ldr)

        RETURN
        END

