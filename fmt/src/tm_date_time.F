	CHARACTER*21 FUNCTION TM_DATE_TIME ()
* V50 - 2/99 *kob* modifications for y2k preparedness.  call 
*                  GET_DATE_AND_TIME which uses FDATE to get
*                  date and time rather than DATE and TIME

	CHARACTER*8	mtime
	CHARACTER*9	mdate
* 2/99 *kob* modifications for y2k preparedness.  call GET_DATE_AND_TIME
*            to get date and time
	CALL GET_DATE_AND_TIME(mdate, mtime)
	TM_DATE_TIME = mdate//' '//mtime
	RETURN
	END
