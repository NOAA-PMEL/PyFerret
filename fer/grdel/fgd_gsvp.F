*
* Set the view fractions for a view transformation
* (view limits in terms of normalized device coordinates)
*

        SUBROUTINE FGD_GSVP(transnum, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER transnum
        REAL*4 xmin, xmax, ymin, ymax

        INCLUDE 'fgrdel.cmn'

*       Sanity checks - transform 0 is implicit identity
        IF ( (transnum .LT. 1) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            STOP 'FGD_GSVP: Invalid transnum value'
        ENDIF
        IF ( (0.0 .GT. xmin) .OR.
     .       (xmin .GE. xmax) .OR.
     .       (xmax .GT. 1.0) ) THEN
            WRITE(*,*) 'FGD_GSVP: xmin = ', xmin, ', xmax = ', xmax
            STOP 'FGD_GSVP: invalid xmin, xmax values'
        ENDIF
        IF ( (0.0 .GT. ymin) .OR.
     .       (ymin .GE. ymax) .OR.
     .       (ymax .GT. 1.0) ) THEN
            WRITE(*,*) 'FGD_GSVP: ymin = ', ymin, ', ymax = ', ymax
            STOP 'FGD_GSVP: invalid ymin, ymax values'
        ENDIF

        viewleftfrac(transnum) = xmin
        viewrightfrac(transnum) = xmax
        viewbottomfrac(transnum) = ymin
        viewtopfrac(transnum) = ymax

        RETURN
        END

