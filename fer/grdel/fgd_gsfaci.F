*
* Set the color to use when creating a temporary brush.
* This unsets the active brush.
* (set fill area color index)
*

        SUBROUTINE FGD_GSFACI(colorindex)

        IMPLICIT NONE
        INTEGER colorindex

        INCLUDE 'fgrdel.cmn'

        lastbrushcolor = colorindex + 1
        activebrush = 0

        RETURN
        END

