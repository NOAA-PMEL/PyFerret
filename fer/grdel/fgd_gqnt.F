*
* Inquire (normalization) transformation
*

        SUBROUTINE FGD_GQNT(transnum, istat, usercoords, viewfracs)

        IMPLICIT NONE
        INTEGER transnum, istat
        REAL*4 usercoords(4), viewfracs(4)

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (transnum .LT. 0) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            STOP 'FGD_GQNT: invalid transnum'
        ENDIF

        IF ( transnum .EQ. 0 ) THEN
*           Implicit transformation - should not actually be used.
*           Using 1000.0 * I instead of just I
            usercoords(1) = 0.0
            usercoords(2) = 1.0
            usercoords(3) = 0.0
            usercoords(4) = 1.0
            viewfracs(1) = 0.0
            viewfracs(2) = 1000.0
            viewfracs(3) = 0.0
            viewfracs(4) = 1000.0
        ELSE
*           Tranformation given in the common block values
            usercoords(1) = viewleftcoord(transnum)
            usercoords(2) = viewrightcoord(transnum)
            usercoords(3) = viewbottomcoord(transnum)
            usercoords(4) = viewtopcoord(transnum)
            viewfracs(1) = viewleftfrac(transnum)
            viewfracs(2) = viewrightfrac(transnum)
            viewfracs(3) = viewbottomfrac(transnum)
            viewfracs(4) = viewtopfrac(transnum)
        ENDIF

        istat = 0
        RETURN
        END

