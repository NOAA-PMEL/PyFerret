*
* Definition of GQPCR
* (inquire predefined color representation)
* Only black and white are automatically set,
* so only set info for those
*

        SUBROUTINE FGD_GQPCR(wstype, colorindex, stat,
     .                 redfrac, greenfrac, bluefrac, opaquefrac)

        IMPLICIT NONE
        INTEGER wstype, colorindex, stat
        REAL*4 redfrac, greenfrac, bluefrac, opaquefrac

*       Opaque white is color index 0, opaque black is color index 1
        IF ( colorindex .EQ. 0 ) THEN
            redfrac = 1.0
            greenfrac = 1.0
            bluefrac = 1.0
            opaquefrac = 1.0
            stat = 0
        ELSE IF ( colorindex .EQ. 1 ) THEN
            redfrac = 0.0
            greenfrac = 0.0
            bluefrac = 0.0
            opaquefrac = 1.0
            stat = 0
        ELSE
            stat = 1
        ENDIF

        RETURN
        END

