*
* Change the name of the default graphics engine and the engine for
* all windows to a no-display engine (Cairo).  This is only called 
* at start-up time before anything has been displayed.
*
        SUBROUTINE FGD_SET_UNMAPPED_DEFAULT()

        IMPLICIT NONE

        INCLUDE 'fgrdel.cmn'

        INTEGER k

        defaultenginename = 'Cairo'
        DO 10 k = 1, maxwindowobjs
            enginename(k) = 'Cairo'
            antialias = .TRUE.
   10   CONTINUE
        RETURN
        END

