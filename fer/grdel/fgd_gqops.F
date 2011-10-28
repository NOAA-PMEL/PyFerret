*
* Defiinition of GQOPS (inquire operating state).
* Just returns 3 (open and running) in its argument
* since there is no explicit engine intitialization
* or finalization.
*

        SUBROUTINE FGD_GQOPS(istat)

        IMPLICIT NONE
        INTEGER istat

        istat = 3

        RETURN
        END

