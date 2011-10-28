*
* Set the hatch style to use when creating a temporary brush.
* This unsets the active brush.
* (set fill area style index)
*

        SUBROUTINE FGD_GSFASI(hstyle)

        IMPLICIT NONE
        INTEGER hstyle

        INCLUDE 'fgrdel.cmn'

        IF ( hstyle .EQ. 1 ) THEN
            lastbrushstyle = "hor"
            lastbstylen = 3
        ELSE IF ( hstyle .EQ. 2 ) THEN
            lastbrushstyle = "fdiag"
            lastbstylen = 5
        ELSE IF ( hstyle .EQ. 3 ) THEN
            lastbrushstyle = "ver"
            lastbstylen = 3
        ELSE IF ( hstyle .EQ. 4 ) THEN
            lastbrushstyle = "bdiag"
            lastbstylen = 5
        ELSE IF ( hstyle .EQ. 5 ) THEN
            lastbrushstyle = "cross"
            lastbstylen = 5
        ELSE IF ( hstyle .EQ. 6 ) THEN
            lastbrushstyle = "diagcross"
            lastbstylen = 9
        ELSE
            STOP 'FGD_GSFASI: Invalid hstyle'
        ENDIF
        activebrush = 0

        RETURN
        END

