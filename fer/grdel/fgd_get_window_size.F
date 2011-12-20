*
* Return the size of the active transform in pixels.
*

        SUBROUTINE FGD_GET_WINDOW_SIZE(width, height)

        IMPLICIT NONE

        REAL width, height

        INCLUDE 'fgrdel.cmn'

        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GET_WINDOW_SIZE: No active window'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GET_WINDOW_SIZE: null active windowobj'
        ENDIF

        width = windowwidth(activewindow) *
     .          windowdpix(activewindow)
        height = windowheight(activewindow) *
     .           windowdpiy(activewindow)

        RETURN
        END

