*
* Set the view fractions for a view transformation
* (view limits in terms of normalized device coordinates)
*

        SUBROUTINE FGD_GSVP(transnum, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER transnum
        REAL*4 xmin, xmax, ymin, ymax

        INCLUDE 'fgrdel.cmn'

        REAL*4 EPSIL, modxmin, modxmax, modymin, modymax
        PARAMETER (EPSIL = 1.0E-4)

*       Sanity checks - transform 0 is implicit identity
        IF ( (transnum .LT. 1) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            STOP 'FGD_GSVP: Invalid transnum value'
        ENDIF

*       Allow some tolerance for floating point error
        IF ( (xmin .LT. 0.0) .AND. (xmin .GT. 0.0-EPSIL) ) THEN
           modxmin = 0.0
        ELSE
           modxmin = xmin
        ENDIF
        IF ( (xmax .GT. 1.0) .AND. (xmax .LT. 1.0+EPSIL) ) THEN
           modxmax = 1.0
        ELSE
           modxmax = xmax
        ENDIF
        IF ( (ymin .LT. 0.0) .AND. (ymin .GT. 0.0-EPSIL) ) THEN
           modymin = 0.0
        ELSE
           modymin = ymin
        ENDIF
        IF ( (ymax .GT. 1.0) .AND. (ymax .LT. 1.0+EPSIL) ) THEN
           modymax = 1.0
        ELSE
           modymax = ymax
        ENDIF

        IF ( (0.0 .GT. modxmin) .OR.
     .       (modxmin .GE. modxmax) .OR.
     .       (modxmax .GT. 1.0) ) THEN
            WRITE(*,*) 'FGD_GSVP: xmin = ', xmin, ', xmax = ', xmax
            STOP 'FGD_GSVP: invalid xmin, xmax values'
        ENDIF
        IF ( (0.0 .GT. modymin) .OR.
     .       (modymin .GE. modymax) .OR.
     .       (modymax .GT. 1.0) ) THEN
            WRITE(*,*) 'FGD_GSVP: ymin = ', ymin, ', ymax = ', ymax
            STOP 'FGD_GSVP: invalid ymin, ymax values'
        ENDIF

        viewleftfrac(transnum) = modxmin
        viewrightfrac(transnum) = modxmax
        viewbottomfrac(transnum) = modymin
        viewtopfrac(transnum) = modymax

        RETURN
        END

