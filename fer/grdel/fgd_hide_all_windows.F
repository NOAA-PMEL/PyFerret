*
* Sets the default visibility of all windows to .FALSE.
*
        SUBROUTINE FGD_HIDE_ALL_WINDOWS()

        IMPLICIT NONE

        INCLUDE 'fgrdel.cmn'

        INTEGER k

        DO 10 k = 1, maxwindowobjs
            showwindow(k) = .FALSE.
  10    CONTINUE

        RETURN
        END

