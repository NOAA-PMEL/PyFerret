*
* Definition of GQLCS
* (inquire locator device state)
* Sets stat to 1 since input functions are not supported at this time
*

        SUBROUTINE FGD_GQLCS(windowid, locdevnum, rettyp, maxdr,
     .                   stat, opmode, echomode, transnum, locx,
     .                   locy, promptyp, echoarea, lendr, datarec)

        IMPLICIT NONE
        INTEGER windowid, locdevnum, rettyp, maxdr, stat, opmode,
     .          echomode, transnum, promptyp, lendr
        REAL locx, locy, echoarea(4)
        CHARACTER*80 datarec
        DIMENSION datarec(maxdr)

       stat = 1

       RETURN
       END

