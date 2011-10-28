*
* Definition of GQASF
* (inquire aspect source flags)
* Sets everything to zero as these flags are not used.
*

        SUBROUTINE FGD_GQASF(stat, lasf)

        IMPLICIT NONE
        INTEGER stat, lasf(13)

        INTEGER k

        stat = 0
        DO 10 k = 1, 13
            lasf(k) = 0
   10   CONTINUE

        RETURN
        END

