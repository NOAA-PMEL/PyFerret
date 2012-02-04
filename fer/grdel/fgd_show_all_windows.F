*
* Sets the default visibility of all windows to .TRUE.
*
        SUBROUTINE FGD_SHOW_ALL_WINDOWS()

        IMPLICIT NONE

        INCLUDE 'fgrdel.cmn'

        INTEGER k

        DO 10 k = 1, maxwindowobjs
            showwindow(k) = .TRUE.
  10    CONTINUE

        RETURN
        END

