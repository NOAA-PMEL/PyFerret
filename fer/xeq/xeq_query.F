    	SUBROUTINE XEQ_QUERY

* execute the QUERY command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* for FERRET V300: 6/16/93
* V301:	11/05/93 - changed QUERY/PIPE to QUERY/IGNORE (a NO-OP)
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
 
#ifdef unix
        include 'tmap_format/tmap_dims.parm'
#       include "tmap_format/tmap_dset.parm"
        include 'tmap_format/xdset_info.cmn_text'
        external xdset_info_data
        include 'tmap_format/xtm_grid.cmn_text'
        external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/slash.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xcommand.cmn'
	include 'ferret_cmn/xgui.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
	INCLUDE 'FERRET_CMN:XGUI.CMN'
#endif

* local parameter declarations:
	INTEGER	    slash_all, slash_file, slash_ignore
	PARAMETER ( slash_all  = 1 + 0,
     .		    slash_file = 1 + 1,
     .		    slash_ignore = 1 + 2 )
	CHARACTER*1 NO_TAG
	PARAMETER ( NO_TAG = " " )

* local variable declarations:
	LOGICAL	do_all, fileout, its_ppl
	INTEGER	LIB_GET_LUN, TM_LENSTR1, STR_UPCASE, vax_code,
     .		chout, flen, qfile, status, i, query, n
	CHARACTER fname*132, qry4*4


* initialize
	fileout = .FALSE.
	status = ferr_ok
        qfile = qual_given( slash_file )

* is this a QUERY/IGNORE NO-OP ?
	IF ( qual_given( slash_ignore ) .GT. 0 ) RETURN

* query given ?
	IF ( num_args .EQ. 0 ) GOTO 5300

* command given with /ALL ?
	do_all = qual_given( slash_all ) .GT. 0

* identify the query - if it starts with 'P' it is a PLOT+ query
	qry4 = cmnd_buff(arg_start(1):arg_end(1))
	vax_code = STR_UPCASE( qry4,
     .		   cmnd_buff(arg_start(1):arg_end(1)) )
	its_ppl = qry4(1:1) .EQ. 'P'
	IF ( .NOT. its_ppl ) THEN
	   DO 10 query = 1, max_queries
	      IF ( queries(query) .EQ. unspecified_name4 ) GOTO 10
	      IF ( qry4 .EQ. queries(query)(:4) )          GOTO 20
 10	   CONTINUE
	   GOTO 5100
	ENDIF

* command given with /FILE  - open output channel
 20        IF ( qfile .GT. 0 ) THEN
	   CALL EQUAL_STR_LC(
     .                  cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .                  fname, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   flen = TM_LENSTR1(fname)
	   IF ( fname.NE.' ' ) THEN
	      status = LIB_GET_LUN( chout )	   
	      OPEN ( UNIT = chout,
     .		     FILE = fname,
     .		     STATUS = 'UNKNOWN',
     .		     ERR = 5200 ) 
	      fileout = .TRUE.
	   ENDIF

	ELSE
	   chout = ttout_lun
	ENDIF

* do it
	IF ( its_ppl ) THEN
!	   QRY_PPL( chout, gui_fmt1, qry4, status ) 	   
	ELSE
	   CALL DO_QUERY( chout, gui_fmt1, query, NO_TAG, status )
	ENDIF
	IF ( status .NE. ferr_ok ) GOTO 5000
	
* exit
 1000	IF ( fileout ) THEN
	   WRITE ( chout, '(40A1)' ) ('=',i=1,40) 
	   CLOSE ( UNIT = chout, ERR = 5000 )
	ENDIF
	RETURN
 3000	FORMAT (I4, 2X, A4)

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		     'unknown QUERY: '//cmnd_buff(arg_start(1):arg_end(1)),
     .		     *1000 )
 5200	CALL ERRMSG( ferr_erreq, status, fname(1:flen), *1000 )
 5300	CALL ERRMSG( ferr_invalid_command, status, 'QUERY what?', *5000 )
	END

