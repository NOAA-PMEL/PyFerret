*
* Sets the window title to be used when creating a window.
*

        SUBROUTINE FGD_GESSPN(wintitle)

        IMPLICIT NONE
        INCLUDE 'fgrdel.cmn'
        CHARACTER*(*) wintitle

        windowtitle = wintitle

        RETURN
        END

