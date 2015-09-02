*
* Assigns the width of outlines for rectangles and other polygons
* in the specified window
*
* Input Arguments:
*     windowid - number of the window to use
*     linewidth - width of outlines; can be zero for no outlines
*
        SUBROUTINE FGD_SET_OUTLINE_WIDTH(windowid, linewidth)

        IMPLICIT NONE
        INTEGER windowid
        REAL*4 linewidth

        INCLUDE 'fgrdel.cmn'

        IF ( (windowid .LT. 1) .OR.
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_SET_OUTLINE_WIDTH: Invalid windowid value'
        ENDIF
        IF ( linewidth .LT. 0.0 ) THEN
            STOP 'FGD_SET_OUTLINE_WIDTH: Invalid linewidth value'
        ENDIF

*       Record the outline width for this window
        outlinewidth(windowid) = linewidth

        RETURN
        END

