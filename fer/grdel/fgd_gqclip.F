*
* Inquire clipping indicator for the active window.
*

        SUBROUTINE FGD_GQCLIP(stat, clipit)

        IMPLICIT NONE
        INTEGER stat, clipit

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR. 
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GQCLIP: Invalid activewindow value'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GQCLIP: null activewindow'
        ENDIF
*       Inquire the clipping state
        IF ( cliptoview(activewindow) ) THEN
            clipit = 1
        ELSE
            clipit = 0
        ENDIF
        stat = 0

        RETURN
        END

