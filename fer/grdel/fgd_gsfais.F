*
* Set the fill style to use when creating a temporary brush.
* This unsets the active brush.
* (set fill area interior style)
*

        SUBROUTINE FGD_GSFAIS(style)

        IMPLICIT NONE
        INTEGER style

        INCLUDE 'fgrdel.cmn'

        IF ( style .EQ. 1 ) THEN
*           Solid brush
            lastbrushstyle = 'solid'
            lastbstylen = 5
        ELSE IF ( style .EQ. 3 ) THEN
*           Hatch brush - default to diagcross
            lastbrushstyle = 'diagcross'
            lastbstylen = 9
        ELSE
            STOP 'FGD_GSFAIS: Invalid style'
        ENDIF
        activebrush = 0

        RETURN
        END

