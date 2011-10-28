*
* Definition of GQCLIP
* Always assigns clipit to 1 as cliiping is always enforced.
*

        SUBROUTINE FGD_GQCLIP(stat, clipit)

        IMPLICIT NONE
        INTEGER stat, clipit

        clipit = 1
        stat = 0

        RETURN
        END

