	LOGICAL FUNCTION TM_DATE_OK( date )

* determine whether a date string is acceptably formatted

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 7/92 *sh* 

* calling argument declarations:
      CHARACTER*(*) date

* internal variable declarations:
      INTEGER	yr, mth, day, hr, min, sec, status

#ifdef unix
        include 'tmap_format/tmap_errors.parm'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

      CALL TM_BREAK_DATE( date,
     .                   yr, mth, day, hr, min, sec, status )
      TM_DATE_OK = status .EQ. merr_ok

	RETURN
	END
