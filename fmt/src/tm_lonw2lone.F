	REAL*8 FUNCTION TM_LONW2LONE( lonw )

* convert a longitude representation which has west positive, discontinuous
* at the dateline (as in EPIC netCDF) to one which has east positive and is
* discontinuous at Greenwhich

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/93

* argument declarations
      REAL*8  lonw

      TM_LONW2LONE = -lonw
      IF ( TM_LONW2LONE .LT. 0 ) TM_LONW2LONE = TM_LONW2LONE + 360.D0

      RETURN
      END
