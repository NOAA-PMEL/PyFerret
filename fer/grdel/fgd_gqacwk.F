*
* Definition of GQACWK (inquire about active workstations)
* Sets retwin to the value of activewindow, and sets
* numactive to one, if activewindow is valid, or zero,
* if activewindow is invalid.  Sets istat to zero and ignores
* nwin.
*

        SUBROUTINE FGD_GQACWK(nwin, istat, numactive, retwin)

        IMPLICIT NONE
        INTEGER nwin, istat, numactive, retwin

        INCLUDE 'fgrdel.cmn'

        INTEGER k

        retwin = activewindow
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            numactive = 0
        ELSE
            numactive = 1
        ENDIF
        istat = 0

        RETURN
        END

