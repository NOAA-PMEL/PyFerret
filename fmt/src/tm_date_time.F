	CHARACTER*21 FUNCTION TM_DATE_TIME ()
	CHARACTER*8	mtime
	CHARACTER*9	mdate
	CALL DATE(mdate)
	CALL TIME(mtime)
	TM_DATE_TIME = mdate(1:7)//'19'//mdate(8:9)//' '//mtime
	RETURN
	END
