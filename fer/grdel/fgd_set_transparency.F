*
* Turns on or off background transparency for images saved on exit.
*
* Input Arguments:
*     transint - if non-zero, save with transparent background;
*                if zero, save with opaque background
*
        SUBROUTINE FGD_SET_TRANSPARENCY(transint)

        IMPLICIT NONE
        INTEGER transint

        INCLUDE 'fgrdel.cmn'

*       Set the value in the common block
        IF ( transint .NE. 0 ) THEN
           savetransparent = .TRUE.
        ELSE
           savetransparent = .FALSE.
        ENDIF

        RETURN
        END

