	SUBROUTINE TM_GET_205_RUN_LINE ( filename, expno, col_head,
     .					 table_row, status )
*
* This routine will read a row from a file storing information about the
* model runs (and maybe other data sets). It will return column heading
* names, the row of experiment information, and a status key.
*
* Arguments:
*	filename  - file to read from
*	expno     - experiment number for desired info
*	col_head  - a string containing column information (Bottom line)
*	table_row - a string containing experiment information
*	status    - return status
*
* Written 04/26/88 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 04/27/88
* revision 0.01 - 02/20/89 - Clarified error message 9100
* revision 0.02 -  1/91 *sh* syntax changes to accomodate DEC RISC compiler
*               - and added TM_INQ_PATH with TMAP_RUNS
*
* IBM Port 14apr94 *kob* : READONLY not allowed in file open statments.  Added
*			   an "#ifdef READONLY_ALLOWED"
* Include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
        include 'tmap_format/xsys_names.cmn_text'
        external xsys_names_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:XSYS_NAMES.CMN'
#endif

* Argument definitions
	CHARACTER*(*)	filename, expno, col_head, table_row
	INTEGER*4	status

* Local Definitions 
	CHARACTER*132	TM_INQ_PATH, buffer
	INTEGER*4	lunit, explen
	INTEGER*4	TM_LENSTR1
	integer		flen	!  for RISC compiler
	character*4	buff4

* for RISC compiler: can't concatenate with CHAR*(*) variable
	flen = MIN( LEN(filename), 132 )	! 132 is buffer size

* locate the data base file (use "buffer" to store the path name temporarily)
        buffer = TM_INQ_PATH( filename, tmap_model_runs, ' ',
     .                        .TRUE., status )
        IF ( status .NE. merr_ok ) GOTO 9999

* Get available logical unit
	CALL LIB_GET_LUN ( lunit )

* Look for filename and open it
	OPEN	( UNIT=lunit,
     .		  FILE=buffer,
     .		  STATUS='OLD',
#ifdef READONLY_ALLOWED
     .		  READONLY,
#endif
     .		  ERR=9000 )

* Now search for the bottom line of the column headings
  100	READ	( lunit, '(A132)', END=9100 ) buffer
	IF ( INDEX(buffer,'=====') .EQ. 0 ) THEN
          col_head = buffer
	  GOTO 100
	ENDIF

* We found the column headings, search for the experiment info
	explen = TM_LENSTR1(expno)
  110	READ	( lunit, '(A132)', END=9200 ) buffer
	IF ( INDEX(buffer(1:10),expno(1:explen)) .EQ. 0 ) GOTO 110
	table_row = buffer
	GOTO 9900

* ERROR - opening filename
 9000	buffer = filename
	CALL TM_ERRMSG ( merr_erreq, status, 'TM_GET_205_RUN_LINE',
     .			 no_descfile, no_stepfile, 'FILE='//buffer(:flen),
     .			 no_errstring, *9999 )

* ERROR - searching for column headings
 9100	buffer = filename
	CALL TM_ERRMSG ( merr_eof, status, 'TM_GET_205_RUN_LINE',
     .			 no_descfile, no_stepfile, 'FILE='//buffer(:flen)//
     .			 '. File not a correct Parameter file.',
     .			 no_errstring, *9990 )

* ERROR - searching for column headings
 9200	buffer = filename
	explen = MIN( explen, 4 )
	buff4 = expno
	CALL TM_ERRMSG ( merr_eof, status, 'TM_GET_205_RUN_LINE',
     .			 no_descfile, no_stepfile,'FILE='//buffer(:flen)//
     .			 ' Experiment "'//buff4(:explen)//'" not found',
     .			 no_errstring, *9990 )

* Status set
 9900	status = merr_ok

* Close file
 9990	CLOSE	( lunit, ERR=9000 )

* Bye bye
 9999	RETURN
	END
