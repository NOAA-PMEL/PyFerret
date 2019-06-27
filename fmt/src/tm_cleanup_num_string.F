	SUBROUTINE TM_CLEANUP_NUM_STRING (string, slen)

* Remove fractional zeros and trailing decimal point from non-exponentials
* Returns string and its new length.


* Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Gropu
*
* 6/24/2019

	INTEGER slen
	CHARACTER*(*) string

	INTEGER nright, idx

        IF ( (INDEX(string, '.') .GT. 0) .AND.
     .       (INDEX(string, 'E') .EQ. 0) ) THEN
           nright = slen
           DO idx = nright,1,-1
              IF (string(idx:idx) .EQ. '0') THEN
                 string(idx:idx) = ' '
                 slen = slen - 1
              ELSE
                 EXIT
              ENDIF
           ENDDO
           IF (string(slen:slen) .EQ. '.') THEN
              string(slen:slen) = ' '
              slen = slen - 1
           ENDIF
        ENDIF

	RETURN
	END