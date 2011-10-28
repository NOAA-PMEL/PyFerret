*
* Deletes (closes and destroys) all Windows created by FGD_GOPWK
* Also deletes any color, brush, font, pen, and symbol objects
* associated with these windows.
*
        SUBROUTINE FGD_GCLKS()

        IMPLICIT NONE
        INCLUDE 'fgrdel.cmn'
        INTEGER k

        DO 10 k = 1, maxwindowobjs
            IF ( windowobjs(k) .NE. nullobj ) THEN
                CALL FGD_GCLWK(k)
            ENDIF
   10   CONTINUE

        RETURN
        END

