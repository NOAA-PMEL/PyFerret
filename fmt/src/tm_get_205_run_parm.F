	SUBROUTINE TM_GET_205_RUN_PARM ( column, col_head, table_row,
     .					 entry, entry_len, status )
*
* This routine will use "column" as a search string in "col_head" to
* locate "entry" in "table_row". "Col_head" and "Table_row" are read
* from a file storing information about the model runs (and maybe other
* data sets).
*
* Arguments:
*	column    - string to search for in col_head
*	col_head  - a string containing column information (Bottom line)
*	table_row - a string containing experiment information
*       entry     - value in table_row pointed to by location of column
*		    in col_head
*	entry_len - length of entry
*	status    - return status
*
* Written 04/26/88 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 04/27/88
* revision 1.00 - 12/02/88 - major changes (unable to document)
* revision 1.01 -  1/91 *sh* syntax changes to accomodate DEC RISC compiler
*
* Include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* Argument definitions
	CHARACTER*(*)	column, col_head, table_row, entry
	INTEGER*4	entry_len, status

* Local Definitions
	CHARACTER*132	up_col,up_head
	INTEGER*4	col_strt, col_end, collen, headlen, i, istat
	INTEGER*4	TM_LENSTR1, STR_UPCASE
	LOGICAL		TM_HAS_STRING

* Initialize
	collen = TM_LENSTR1(column)
	headlen = TM_LENSTR1(col_head)
	status = STR_UPCASE(up_col,column(1:collen))
	status = STR_UPCASE(up_head,col_head(1:headlen))

* Locate "column" in "col_head"
	IF ( .NOT. TM_HAS_STRING(up_head(1:headlen),column(1:collen)) )
     .		 GOTO 9000

* Found it, let's get the column start and end locations
	DO i=INDEX(up_head(1:headlen),up_col(1:collen)),1,-1
	  IF (up_head(i:i) .EQ. ' ') THEN
	    col_strt = i+1
	    GOTO 100
	  ENDIF
	ENDDO
	col_strt = 1
  100	col_end  = INDEX(up_head(col_strt:),' ') + col_strt-2

* Pull out the entry now
	IF (col_end .GT. col_strt) entry = table_row(col_strt:col_end)

* Left justify, and determine length of entry
	DO i=1,LEN(entry)
	  IF (entry(i:i) .NE. ' ' .AND. entry(i:i) .NE. '	') THEN
	    entry = entry(i:)
	    entry_len = TM_LENSTR1(entry)
	    GOTO 9900
	  ENDIF
	ENDDO
	entry_len = 1
	GOTO 9900

* ERROR - searching for "column" in column headings
 9000	up_col = Column		! RISC bug: cannot use CHAR*(*) in concat
	CALL TM_ERRMSG ( merr_keynf, status, 'TM_GET_205_RUN_PARM',
     .			 no_descfile, no_stepfile, 'Column='//up_col(:collen),
     .			 no_errstring, *9999 )

* Status set
 9900	status = merr_ok

* Bye bye
 9999	RETURN
	END
