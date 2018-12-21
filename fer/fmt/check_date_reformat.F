	SUBROUTINE CHECK_DATE_REFORMAT (datestring, slen, cal_id, status)


* For option set by MODE DATE_FORMAT to write dates in ISO8601 format
* See if the output-format stored in the program setting date_fmt_out
* is pdate_numslashnum, and if so rewrite the date string in that form.
*
* V7.44+  Ansley Manke 2018-12-12
*         See GitHub issue #1909
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd

        include 'tmap_dims.parm'
        include 'tmap_errors.parm'
        include 'ferret.parm'
        include 'xprog_state.cmn'


	CHARACTER*(*)	datestring
	INTEGER		slen, cal_id, status

	LOGICAL		do_err
	CHARACTER*35	buff
	INTEGER		TM_LENSTR1
	
	do_err = .FALSE.
	
	IF (date_fmt_out .EQ. pdate_numslashnum) THEN 
	   CALL TM_DATE_REFORMAT( datestring, pdate_vax, cal_id,
     .				  buff, date_fmt_out, do_err, status)
	   IF (status .EQ. merr_ok) THEN
	      slen = TM_LENSTR1 (buff)
	      datestring = buff
	   ENDIF
	ENDIF

	RETURN
	END
