*
* Definition of GRQCH
* (request choice)
* Only sets outstat to 0 as input functions are not supported at this time.
*

        SUBROUTINE FGD_GRQCH(windowid, chdevnum, outstat, outchnum)

        IMPLICIT NONE
        INTEGER windowid, chdevnum, outstat, outchnum

        outstat = 0

        RETURN
        END

