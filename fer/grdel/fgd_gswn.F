*
* Set the user (world) coordinates for a view transformation
*

        SUBROUTINE FGD_GSWN(transnum, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER transnum
        REAL*4 xmin, xmax, ymin, ymax

        INCLUDE 'fgrdel.cmn'

*       Sanity checks - transform 0 is implicit identity
        IF ( (transnum .LT. 1) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            STOP 'FGD_GSWN: Invalid transnum value'
        ENDIF
        IF ( (xmax - xmin) .LE. 0.0 ) THEN
            WRITE(*,*) 'FGD_GSWN: xmin = ', xmin, ', xmax = ', xmax
            STOP 'FGD_GSWN: invalid xmin, xmax values'
        ENDIF
        IF ( (ymax - ymin) .LE. 0.0 ) THEN
            WRITE(*,*) 'FGD_GSWN: ymin = ', ymin, ', ymax = ', ymax
            STOP 'FGD_GSWN: invalid ymin, ymax values'
        ENDIF

        viewleftcoord(transnum) = xmin
        viewrightcoord(transnum) = xmax
        viewbottomcoord(transnum) = ymin
        viewtopcoord(transnum) = ymax

        RETURN
        END

