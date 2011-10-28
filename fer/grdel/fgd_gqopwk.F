*
* Definition of GQOPWK (inquire about open workstations)
* Just returns the number of non-null window objects in
* the numopen argument, and the window ID number of the
* first of these in the retwin argument.  The istat argument
* is set to zero, and the nwin argument is ignored.
*

        SUBROUTINE FGD_GQOPWK(nwin, istat, numopen, retwin)

        IMPLICIT NONE
        INTEGER nwin, istat, numopen, retwin

        INCLUDE 'fgrdel.cmn'

        INTEGER k

        numopen = 0
        retwin = 0
        DO 10 k = 1, maxwindowobjs
            IF ( windowobjs(k) .NE. nullobj ) THEN
                numopen = numopen + 1
                IF ( retwin .EQ. 0 ) retwin = k
            ENDIF
   10   CONTINUE
        istat = 0

        RETURN
        END

