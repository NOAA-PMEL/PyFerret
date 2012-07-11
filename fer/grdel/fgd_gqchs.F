*
* Definition of GQCHS
* (inquire choice device state
* Sets stat to 1 since input functions are not supported at this time
*

        SUBROUTINE FGD_GQCHS(windowid, chdevnum, maxdr, stat,
     .                   opmode, echomode, chnum, promptyp,
     .                   echoarea, lendr, datarec)

        IMPLICIT NONE
        INTEGER windowid, chdevnum, maxdr, stat, opmode,
     .          echomode, chnum, promptyp, lendr
        REAL*4 echoarea(4)
        CHARACTER*80 datarec
        DIMENSION datarec(maxdr)

       stat = 1

       RETURN
       END

