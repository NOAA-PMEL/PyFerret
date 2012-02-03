*
* Return arguments:
*    inanimation: 1 if the current plotting is part of an animation
*                   and is not the first frame;
*                 0 otherwise.
*

        SUBROUTINE FGD_GETANIMATE(inanimation)

        IMPLICIT NONE
        INTEGER inanimation
        INCLUDE 'xplot_setup.cmn'

        IF ( .NOT. animate ) THEN
            inanimation = 0
        ELSE IF ( first_frame ) THEN
            inanimation = 0
        ELSE
            inanimation = 1
        ENDIF

        RETURN
        END

