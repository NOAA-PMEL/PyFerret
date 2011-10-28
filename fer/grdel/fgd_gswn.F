*
* Set the user (world) coordinates for a view transformation
*

        SUBROUTINE FGD_GSWN(transnum, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER transnum
        REAL xmin, xmax, ymin, ymax

        INCLUDE 'fgrdel.cmn'

*       Sanity checks - transform 0 is implicit identity
        IF ( (transnum .LT. 1) .OR.
     .       (transnum .GT. maxtransforms) ) THEN
            STOP 'FGD_GSWN: Invalid transnum value'
        ENDIF
*       Since the world coordinates may be treated as
*       integer values make sure there is significant
*       difference between the min and max values.
        IF ( (xmax - xmin) .LT. 5 ) THEN
            STOP 'FGD_GSWN: invalid xmin, xmax values'
        ENDIF
        IF ( (ymax - ymin) .LT. 5 ) THEN
            STOP 'FGD_GSWN: invalid ymin, ymax values'
        ENDIF

        viewleftcoord(transnum) = xmin
        viewrightcoord(transnum) = xmax
        viewbottomcoord(transnum) = ymin
        viewtopcoord(transnum) = ymax

        RETURN
        END

