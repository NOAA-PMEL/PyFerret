	SUBROUTINE SAMPLE( memory, mr_list, cx_list, nvars,
     .			   opt1_string, opt2_string, file_given,
     .			   fname, fmt, binary, columns, status )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* sample the given field at the specified XYZT coordinates
* opt1_string contains special instructions
*	"FILL" indicates to fill holes when the field contains missing values
* opt2_string contains flags to indicate if the coordinates are given as
*	world positions (C=Coordinate) or subscripts (I=index)

*  USER/COMMAND=SAMPLE/OPT1=FILL,STANDARD_BAD/OPT2=CCII/FILE=outfile/FORM=(F10)

* The format may be a FORTRAN FORMAT or "UNFORMATTED", or,
* "UNFORMATTED:columns" - the latter produces condensed output (results, only)

* Each output record will consist of
*	value xcoord ycoord zcoord tcoord code
* where "code" is 0=fully successful
*		  1=successful with hole filling
*              -1-4=failed due to beyond input data limits on axis -code
*		 -9=failed due to hole in input data field
*		-99=failed due to hole in input coordinates

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V301:  1/94
*
* V312: 5/94 - array "memory" as a calling argument
* V314: 8/22/94 *kob* IBM port: 1) ACCESS=APPEND not allowed in file opens - added
*				   ifdef NO_ACCESS_APPEND check
*				2) TM_HAS_STRING incorrectly defined as INTEGER
*				3) inherited length string concats are not legal -
*					use risc buff
* Linux Port 5/97 *kob* - had to add ifdef F90_OPEN_FILE_APPEND because f90
*			  uses "POSITION" instead of "ACCESS" to specify 
*			  opening a file for append.

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
#ifdef NO_PASSED_CONCAT	
	include 'xrisc.cmn'
#endif
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	LOGICAL	file_given, binary
	INTEGER	nvars, mr_list(nvars), cx_list(nvars), columns, status
	REAL	memory( mem_blk_size, max_mem_blks )
	CHARACTER opt1_string*(*), opt2_string*(*), fname*(*), fmt*(*)

* internal variable declarations:
	LOGICAL	fill, standard_bad, tempx, tempy, tempz, tempt,
     .		TM_HAS_STRING
	INTEGER	TM_LENSTR1, CGRID_SIZE, STR_UPCASE,
     .		CX_DIM_LEN,
     .		mblk1, nblks, col_mblk1, col_nblks, slen,
     .		ssorww(4), out_lun,
     .		i, flen, fmlen, npts, cx, ndim, dim(4), cxpts(2:5)
	REAL	missing
	CHARACTER VAR_TRANS*48, buff*48
	
* internal parameter definitions
	INTEGER by_ss, by_ww, by_nothin
	PARAMETER ( by_ss=1, by_ww=2, by_nothin=0 )

* DECODE THE OPT1 COMMAND QUALIFIER:
* /OPT1 = "FILL,STANDARD_BAD"
	slen = TM_LENSTR1( opt1_string )
	fill = TM_HAS_STRING(opt1_string(:slen),'FILL')
	standard_bad = TM_HAS_STRING(opt1_string(:slen),'STANDARD')
	IF ( .NOT.fill .AND. .NOT.standard_bad .AND. opt1_string.NE.' ' 
     .	.AND. .NOT.TM_HAS_STRING(opt1_string(:slen),'HOLE') ) GOTO 5100

* Must be exactly 5 arguments given
* arguments 2-5 must be either same size or constants
	IF ( nvars .NE. 5 ) GOTO 5200
	npts = unspecified_int4
	DO 100 i = 2, 5
	   cx = cx_list(i)
	   CALL GET_CX_DIMS( cx, ndim, dim ) 
	   cxpts(i) = CGRID_SIZE( cx )	
	   IF ( cxpts(i) .NE. 1 ) THEN
	      IF ( npts .EQ. unspecified_int4 ) npts = cxpts(i)
	      IF ( npts .NE. CGRID_SIZE( cx ) ) GOTO 5300
	   ENDIF
 100	CONTINUE
	IF ( npts .EQ. unspecified_int4 ) npts = 1
	
* determine what to output for missing data points
	IF ( standard_bad ) THEN
	   missing = bad_val4
	ELSE
	   missing = mr_bad_data(mr_list(1))
	ENDIF

* Decode the specification of subscript or world coordinate
* /OPT2 = "C" (all coordinates) or "I" (all indices) or "CCIN", ...
	DO 200 i = 1, 4
 200	ssorww(i) = by_ww			! default to "C"
	IF ( opt2_string .NE. ' ' ) THEN
	   i = STR_UPCASE( opt2_string, opt2_string )
	   slen = TM_LENSTR1( opt2_string )
	   IF ( slen.NE.1 .AND. slen.NE.4 ) GOTO 5400
	   IF ( slen.EQ.1 ) THEN     ! replicate single char shorthand
	      DO 210 i = 2, 4
 210	      opt2_string(i:i) = opt2_string(1:1)
	   ENDIF
	   DO 220 i = 1, 4
	      IF ( opt2_string(i:i) .EQ. 'C' ) THEN
	         ssorww(i) = by_ww
	      ELSEIF ( opt2_string(i:i) .EQ. 'I' ) THEN
	         ssorww(i) = by_ss
	      ELSEIF ( opt2_string(i:i) .EQ. 'N' ) THEN
	         ssorww(i) = by_nothin
	      ELSE
	         GOTO 5400
	      ENDIF
 220	   CONTINUE
	ENDIF

* narrow the ssorww choices if only a bad_flag was given for a coord
	DO 250 i = 2,5
	   IF ( cxpts(i).EQ.1 .AND. ( memory(1,mr_blk1(mr_list(i)))
     .		.EQ. mr_bad_data(mr_list(i)) )  ) ssorww(i-1) = by_nothin
 250	CONTINUE

* make sure the field of data provided doesn't create ambiguities on the
* unsampled axes  (i.e. multiple points provided but no sampling coordinates)
	cx = cx_list(1)
	DO 300 i = 1, 4
	   IF ( ssorww(i) .EQ. by_nothin
     .	   .AND. CX_DIM_LEN(i, cx) .GT. 1 ) GOTO 5500
 300	CONTINUE	

* make sure there is no attempt to sample on a normal axis
	DO 400 i = 1, 4
	   IF ( ssorww(i) .NE. by_nothin
     .	   .AND. cx_lo_ss(cx,i) .EQ. unspecified_int4 ) GOTO 5700
 400	CONTINUE	

* insert the default filename based on format type
* or prepare for terminal output
	IF ( file_given ) THEN
	   IF ( fname .EQ. ' ' ) THEN
	      IF ( binary ) THEN
	         fname = 'fer_sample.unf'
	      ELSE
	         fname = 'fer_sample.dat'
	      ENDIF
	   ENDIF
	ENDIF
	IF ( .NOT.binary .AND. fmt.EQ.' ' )
     .			fmt = '(5(1PG14.6),0PF5.0)'   ! "0P" for DEC bug
	flen = TM_LENSTR1( fname )
	fmlen = TM_LENSTR1( fmt )

* allocate memory for consolidated 4D coordinates
	CALL GET_MEMORY( npts*4, mblk1, nblks, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* allocate memory for output record buffer if needed
	IF ( columns .GT. 0 ) THEN
	   CALL GET_MEMORY( columns, col_mblk1, col_nblks, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSE
	   col_mblk1 = 1	! dummy, unused
	ENDIF

* Open the output file
	IF ( file_given ) THEN
	   IF ( fmt .EQ. ' ' ) THEN
	      OPEN( UNIT        = default_list_lun,
     .              FILE        = fname,
     .              FORM        = 'UNFORMATTED',
#ifdef NO_ACCESS_APPEND	
#else
#ifdef F90_OPEN_FILE_APPEND
     .		    POSITION    = 'APPEND',
#else
     .              ACCESS      = 'APPEND',
#endif
#endif
     .              STATUS      = 'UNKNOWN',
     .              ERR         = 5600 )
!	      WRITE (ttout_lun, 3000) 'binary', fname(:flen)
	   ELSE
	      OPEN( UNIT        = default_list_lun,
     .              FILE        = fname,
     .              FORM        = 'FORMATTED',
#ifdef NO_ACCESS_APPEND	
#else
#ifdef F90_OPEN_FILE_APPEND
     .		    POSITION    = 'APPEND',
#else
     .              ACCESS      = 'APPEND',
#endif
#endif
     .              STATUS      = 'UNKNOWN',
     .              ERR         = 5600 )
!	      WRITE (ttout_lun, 3000) 'ASCII', fname(:flen)
	   ENDIF
! 3000	   FORMAT (' Sampled output is in ',A,' file ',A)
	   out_lun = default_list_lun
	ELSE
	   out_lun = ttout_lun
	ENDIF

* call the SAMPLing work routine
	CALL SAMPLE_SUB( memory, cx_list(2), mr_list(2), npts, cxpts,
     .	memory(1, mr_blk1(mr_list(1))), mr_list(1), cx_list(1),
     .	missing,
     .	memory(1, mblk1), memory(1, col_mblk1), 
     .  fmt(1:fmlen), fill, ssorww,
     .	out_lun, binary, columns, status )
     
* close the output file
	IF ( file_given ) CLOSE(UNIT=out_lun, ERR=5600)

* cleanup
 1000	CALL FREE_MEMORY( mblk1, nblks )
	IF ( columns .GT. 0 ) THEN
	   CALL FREE_MEMORY( col_mblk1, col_nblks )
	ENDIF
	RETURN

* error exits
 5000	RETURN
#ifdef NO_PASSED_CONCAT
 5100	risc_buff = opt1_string(:slen)
	CALL ERRMSG( ferr_invalid_command, status,
     .		    '/OPT1='//risc_buff//pCR//
     .		    'May be "FILL" or "HOLES" with "STANDARD_BAD"',
     .		     *5000 )
#else
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		    '/OPT1='//opt1_string(:slen)//pCR//
     .		    'May be "FILL" or "HOLES" with "STANDARD_BAD"',
     .		     *5000 )
#endif
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		    'arguments must be field,xpts,ypts,zpts,tpts', *5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		    '# of X,Y,Z, and T coordinates must each be same or 1',
     .		    *5000 )
#ifdef NO_PASSED_CONCAT
 5400   risc_buff = opt2_string(:slen)
	CALL ERRMSG( ferr_invalid_command, status,
     .   '/OPT2='//risc_buff//pCR//
     .   'Must be xxxx - 4 letters of "C"(coord),'
     .   //' "I"(index) or "N"(N/A)',
     .		    *5000 )
#else
 5400	CALL ERRMSG( ferr_invalid_command, status,
     .   '/OPT2='//opt2_string(:slen)//pCR//
     .   'Must be xxxx - 4 letters of "C"(coord),'
     .   //' "I"(index) or "N"(N/A)',
     .		    *5000 )
#endif
 5500	buff = VAR_TRANS(i, cx, slen )
	CALL ERRMSG( ferr_invalid_command, status,
     .  'Sampling of '//buff(:slen)//' is ambiguous on '//
     .	ww_dim_name(i)//' axis', *5000 )
 5600	CALL ERRMSG( ferr_erreq,status,fname,*1000 )
 5700	buff = VAR_TRANS(i, cx, slen )
	CALL ERRMSG( ferr_invalid_command, status,
     .  'Attempt to sample normal '//ww_dim_name(i)//' axis of '//
     .  buff(:slen), *5000 )

	END

