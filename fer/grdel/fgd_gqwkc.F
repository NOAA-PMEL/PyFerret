*
* Definition of GQWKC
* (inquire workstatrion connection and type)
*

        SUBROUTINE FGD_GQWKC(windowid, istat, connectid, wstype)

        IMPLICIT NONE
        INTEGER windowid, istat, connectid, wstype

        INCLUDE 'fgrdel.cmn'

*       Sanity check
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GQWKC: Invalid windowid value'
        ENDIF

*       Canned return values based of XWindows display
        connectid = 1
        wstype = 4
        istat = 0

        RETURN
        END
