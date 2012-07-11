*
* Define the normalized device coordinates limits for the full window.
* (set workstation window)
* In other words: the full window, instead of
* going from zero to one for X and Y, 
* it goes from xmin to xmax for X 
*          and ymin to ymax for Y
* Ferret always sets xmin and ymin to zero,
* and either xmax or ymax to one, with the other to (0.0,1.0],
* thus used to adjust for the aspect ratio of the monitor.
*

        SUBROUTINE FGD_GSWKWN(windowid, xmin, xmax, ymin, ymax)

        IMPLICIT NONE
        INTEGER windowid
        REAL*4 xmin, xmax, ymin, ymax

        INCLUDE 'fgrdel.cmn'

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GSWKWN: Invalid windowid value'
        ENDIF
        IF ( xmin .NE. 0.0 ) THEN
            STOP 'FGD_GSWKWN: Unexpected non-zero xmin'
        ENDIF
        IF ( ymin .NE. 0.0 ) THEN
            STOP 'FGD_GSWKWN: Unexpected non-zero ymin'
        ENDIF
        IF ( xmax .GT. 1.0 ) THEN
            STOP 'FGD_GSWKWN: Unexpected xmax greater than one'
        ENDIF
        IF ( ymax .GT. 1.0 ) THEN
            STOP 'FGD_GSWKWN: Unexpected ymax greater than one'
        ENDIF

        ndcxmax(windowid) = xmax
        ndcymax(windowid) = ymax

        RETURN
        END

