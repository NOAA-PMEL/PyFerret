*
* Definition of GRQLC
* (request locator)
* Assigns outstat to 0 since input functions are not supported at this time.
*

        SUBROUTINE FGD_GRQLC(windowid, locdevnum,
     .                   outstat, transnum, outx, outy)

        IMPLICIT NONE
        INTEGER windowid, locdevnum, outstat, transnum
        REAL*4 outx, outy

        outstat = 0

        RETURN
        END

