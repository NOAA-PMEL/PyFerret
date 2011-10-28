*
* Definition of GPREC (pack data record)
* Does nothing.
*
        SUBROUTINE FGD_GPREC(il, ia, rl, ra, sl, lstr, str, mldr,
     .                   errind, ldr, datarec)

        IMPLICIT NONE
        INTEGER il, ia, rl, sl, lstr, mldr, errind, ldr
        REAL ra
        CHARACTER*10 str
        CHARACTER*80 datarec
        DIMENSION ia(il), ra(rl), lstr(sl), str(sl), datarec(mldr)

        RETURN
        END

