	SUBROUTINE EZ_READ ( dset, nread, ncol, columns, buff, status,
     .	                      d0, d1, d2, d3, d4,
     .	                      d5, d6, d7, d8, d9,
     .	                      d10,d11,d12,d13,d14,
     .	                      d15,d16,d17,d18,d19  )
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
* 
* Subroutine uses the COMMON info for the initalized EZ data set (ASCII and
* unformatted non_TMAP data files) to read in the entire data file and transfer 
* it to the variable arrays d0 .. d19.
*
* revision 0.00 - 08/04/89
* revision 1.00 - 08/15/89 - Will initialize XDSET/XSTEP common variables as
*			     necessary (first time, and if dset_mod is set)
* revision 1.10 - 10/24/89 - *sh* eliminated EZ_UPDATE_COMMON call
*			     added multi-column reads of same variable
* revision 1.20 - 02/27/90 - Improved read error message by including record #
* RISC/Unix port *sh* 1/91 - syntax changes to accomodate DEC RISC compiler
*                     5/91 - corrected bug sf_valid(dset) --> sf_valid(sfcnt)
* 3/12/93 *sh* : made "buff" into an argument in the call
* IBM/AIX Port ---
*       4/13/94 *kob* : Put definitions of cr and lf into Parameter statments
*                     : Added "#ifdef READONLY_ALLOWED" because AIX compiler
*                       didn't allow readonly in open statements
* 5/94: *sh* major change to permit /ORDER= control over input
*       converted internal logic to 1D (ncol*nread) instead of 2D (ncol,nread) 
*	added /FORMAT=STREAM, as well, where the file is opened for direct
*	access in longwords, /SKIP signifies the number of words to skip,
*	and /COLUMNS signifies the number of words separating multiple
*	variables in the file
* 2/95:  increased from 10 arguments to 20
* 9/95: *sh* & *kob* - changed error branch for STREAM reads
* 9/95	kob*	- added ifdef for sgi to prevent setting reclen in words 
*		  instead of bytes
* 11/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.

* Arguments:
*	dset	- data set number
*	nread	- (input/output) number of data records to read/actually read
*	ncol    - number of values PER VARIABLE per record ("ngroups" in PLOT+)
*	columns - number of VALUES per record
*       buff    - buffer big enough to hold one record
*	status	- (output)
*	d0...d19- (output) variable arrays
*
	INTEGER*4	dset, nread, ncol, columns, status, nmax
	REAL*4		d0(ncol*nread), d1(ncol*nread), d2(ncol*nread),
     .			d3(ncol*nread), d4(ncol*nread), d5(ncol*nread), 
     .			d6(ncol*nread), d7(ncol*nread), d8(ncol*nread), 
     .			d9(ncol*nread),
     .			d10(ncol*nread),d11(ncol*nread),d12(ncol*nread),
     .                  d13(ncol*nread),d14(ncol*nread),d15(ncol*nread),
     .                  d16(ncol*nread),d17(ncol*nread),d18(ncol*nread),
     .                  d19(ncol*nread)
        REAL*4 buff(columns)

*
* Include files
#ifdef unix
	include 'tmap_format/ez_lib.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xez_info.cmn_text'
	external xez_info_data
	include 'tmap_format/xez_order.cmn_text'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:EZ_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XEZ_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XEZ_ORDER.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif
*
*internal parameter declarations *kob* 4apr94
        CHARACTER*1     cr,lf
        PARAMETER       ( cr = CHAR(13),
     .                    lf = CHAR(10))
* Local definitions
	CHARACTER*11	file_format
	CHARACTER*13	TM_STRING
	CHARACTER*80	errbuff
	CHARACTER*99	in_format
	INTEGER		EZ_PERMUTED_INDEX
	INTEGER*4	tpos, vpos, vcnt, sfcnt, lunit, scnt, rcnt, vfirst
	INTEGER*4	skip_in, read_in, transloc, icol, nout, ncolmn
	INTEGER*4	itemp, i, initloc, irec, ierr
	INTEGER*4	trans(0:maxezvars-1)
	DATA 		(trans(i),i=0,maxezvars-1) /maxezvars*-1/
	LOGICAL		unformatted, file_exists, file_open,
     .			permuted, stream
	REAL		bad(0:maxezvars-1)

* Does data set exist?
	IF (ds_name(dset) .EQ. char_init30) GOTO 9300
*
* Locate Step file entry
	DO 10 sfcnt = 1,maxstepfiles
	  IF (sf_setnum(sfcnt) .EQ. dset) GOTO 20
  10	CONTINUE
	GOTO 9400

* Initialize
 20	nmax = ncol*nread
*   get file format
	stream = ds_parm_text(1,dset)(:3) .EQ. 'STR'
	unformatted = ds_parm_text(1,dset)(:3) .EQ. 'UNF'
	IF ( unformatted ) THEN
	  file_format = 'UNFORMATTED'
	ELSE
	  file_format = 'FORMATTED'
	  in_format = ds_parm_text(1,dset)		! Run time format info
	ENDIF
*   load transfer array. vpos will end up with number of variables to read in
*   from file, tpos will be number of useful variables to transfer to variable
*   arrays. '-' in ds_var_code indicates unused variable
*   4/94 - also capture the var # of the first var in the data set, vfirst
	tpos = 0
	vpos = 0
	DO 100 vcnt = 1,maxvars
	  IF (ds_var_setnum(vcnt) .EQ. dset) THEN
	    IF ( vpos .EQ. 0 ) vfirst = vcnt
	    vpos = vpos + 1
	    IF (Index(ds_var_code(vcnt),'-') .EQ. 0) THEN
	      trans(tpos) = vpos
	      tpos = tpos + 1
	      bad(tpos) = ds_bad_flag(vcnt)    ! 4/94
	    ENDIF
	  ENDIF
  100	CONTINUE

* is the input permuted?
	permuted = .FALSE.
	DO 110 i = 1, 4
	   IF ( ds_ordering(i,vfirst) .NE. i ) permuted = .TRUE.
 110	CONTINUE
!	permuted = .TRUE.   ! debug statement
	IF ( permuted ) CALL EZ_INIT_PERMUTE( vfirst )

*   set up ASSIGN statements (for GOTO variable statements)
	IF ( unformatted ) THEN
	  ASSIGN 400 TO skip_in			! Goes to UNFORMATTED read
	  ASSIGN 500 TO read_in			! Goes to UNFORMATTED read
	ELSEIF ( in_format(:4) .EQ. 'FREE' ) THEN
	  ASSIGN 450 TO skip_in			! Goes to FORMATTED read
	  ASSIGN 610 TO read_in			! Goes to FORMATTED read
	ELSEIF ( stream ) THEN			! "STREAM" direct access binary
	  ASSIGN 499 TO skip_in
	  ASSIGN 620 TO read_in
	  irec = sf_skip(sfcnt) + 1     ! /SKIP=words
	ELSE
	  ASSIGN 450 TO skip_in			! Goes to FORMATTED read
	  ASSIGN 600 TO read_in			! Goes to FORMATTED read
	ENDIF
	GOTO (150 ,151 ,152 ,153 ,154 ,155 ,156 ,157 ,158 ,159,
     .	      1510,1511,1512,1513,1514,1515,1516,1517,1518,1519) tpos
	ASSIGN 799 TO transloc			! transfers 0 variables
	ASSIGN 390 TO initloc
	GOTO 160
  150	ASSIGN 790 TO transloc			! transfers 1 variable
	ASSIGN 380 TO initloc
	GOTO 160
  151	ASSIGN 780 TO transloc			! transfers 2 variables
	ASSIGN 381 TO initloc
	GOTO 160
  152	ASSIGN 770 TO transloc			! transfers 3 variables
	ASSIGN 382 TO initloc
	GOTO 160
  153	ASSIGN 760 TO transloc			! transfers 4 variables
	ASSIGN 383 TO initloc
	GOTO 160
  154	ASSIGN 750 TO transloc			! transfers 5 variables
	ASSIGN 384 TO initloc
	GOTO 160
  155	ASSIGN 740 TO transloc			! transfers 6 variables
	ASSIGN 385 TO initloc
	GOTO 160
  156	ASSIGN 730 TO transloc			! transfers 7 variables
	ASSIGN 386 TO initloc
	GOTO 160
  157	ASSIGN 720 TO transloc			! transfers 8 variables
	ASSIGN 387 TO initloc
	GOTO 160
  158	ASSIGN 710 TO transloc			! transfers 9 variables
	ASSIGN 388 TO initloc
	GOTO 160
  159	ASSIGN 700 TO transloc			! transfers 10 variables
	ASSIGN 389 TO initloc
	GOTO 160
 1510	ASSIGN 7910 TO transloc			! transfers 11 variable
	ASSIGN 3810 TO initloc
	GOTO 160
 1511	ASSIGN 7810 TO transloc			! transfers 12 variables
	ASSIGN 3811 TO initloc
	GOTO 160
 1512	ASSIGN 7710 TO transloc			! transfers 13 variables
	ASSIGN 3812 TO initloc
	GOTO 160
 1513	ASSIGN 7610 TO transloc			! transfers 14 variables
	ASSIGN 3813 TO initloc
	GOTO 160
 1514	ASSIGN 7510 TO transloc			! transfers 15 variables
	ASSIGN 3814 TO initloc
	GOTO 160
 1515	ASSIGN 7410 TO transloc			! transfers 16 variables
	ASSIGN 3815 TO initloc
	GOTO 160
 1516	ASSIGN 7310 TO transloc			! transfers 17 variables
	ASSIGN 3816 TO initloc
	GOTO 160
 1517	ASSIGN 7210 TO transloc			! transfers 18 variables
	ASSIGN 3817 TO initloc
	GOTO 160
 1518	ASSIGN 7110 TO transloc			! transfers 19 variables
	ASSIGN 3818 TO initloc
	GOTO 160
 1519	ASSIGN 7010 TO transloc			! transfers 20 variables
	ASSIGN 3819 TO initloc
*
* Check file validity if necessary
  160	IF (.NOT. sf_valid(sfcnt)) THEN
!	  INQUIRE (	FILE = sf_name(sfcnt),	! sf_name is too short 10/89
	  INQUIRE (	FILE = ds_des_name(dset),
     .			RECL = itemp,	! temporary var needed by DEC/RISC
     .			EXIST = file_exists,
     .			OPENED = file_open,
     .			ERR = 9000)
	  IF (.NOT. file_exists) GOTO 9100
	  sf_reclen(sfcnt) = itemp
* calculate RECL in longwords for unformatted (if not open)             
          IF (.NOT. file_open .AND. unformatted )
     .	                         sf_reclen(sfcnt) = sf_reclen(sfcnt)/4
	  sf_valid(sfcnt) = .TRUE.      ! *sh* dset -> sfcnt 5/91
        ENDIF                                                              
*
* get free VMS logical unit #   
	CALL LIB_GET_LUN (lunit)
*
* open the file
#ifdef sun 
#  ifndef sgi
#    define reclen_in_bytes
#  endif
#endif
	IF ( stream ) THEN       ! 5/94
	   OPEN	(UNIT = lunit,
     .		 FILE = ds_des_name(dset),
     .		 FORM = 'UNFORMATTED',
     .		 ACCESS = 'DIRECT',
#ifdef reclen_in_bytes
     .		 RECL = 4,
#else
     .		 RECL = 1,
#endif
     .		 STATUS = 'OLD',
     .		 ERR = 9000)
	ELSE
* added ifdef for READONLY_ALLOWED for IBM port *kob* 4apr94
	   OPEN	(UNIT = lunit,
     .		 FILE = ds_des_name(dset),
     .		 FORM = file_format,
     .		 ACCESS = 'SEQUENTIAL',
     .		 STATUS = 'OLD',
#ifdef READONLY_ALLOWED
     .		 READONLY,
#endif
     .		 ERR = 9000)
	ENDIF
	sf_lunit(sfcnt) = lunit    ! *sh* dset -> sfcnt 5/91
*
* if input is to be permuted then it is essential to pre-fill the arrays
* that will receive the data - this because a fragmentary read (less data in
* the file then the grid dimensions would imply ) will end
* at a point in the middle of the output arrays.
	IF ( permuted ) THEN
	   DO 390 nout = 1, nmax
	      GOTO initloc
 3819	      d19(nout) = bad(trans(19))	      
 3818	      d18(nout) = bad(trans(18))	      
 3817	      d17(nout) = bad(trans(17))	      
 3816	      d16(nout) = bad(trans(16))	      
 3815	      d15(nout) = bad(trans(15))	      
 3814	      d14(nout) = bad(trans(14))	      
 3813	      d13(nout) = bad(trans(13))	      
 3812	      d12(nout) = bad(trans(12))	      
 3811	      d11(nout) = bad(trans(11))	      
 3810	      d10(nout) = bad(trans(10))	      
 389	      d9(nout) = bad(trans(9))	      
 388	      d8(nout) = bad(trans(8))	      
 387	      d7(nout) = bad(trans(7))	      
 386	      d6(nout) = bad(trans(6))	      
 385	      d5(nout) = bad(trans(5))	      
 384	      d4(nout) = bad(trans(4))	      
 383	      d3(nout) = bad(trans(3))	      
 382	      d2(nout) = bad(trans(2))	      
 381	      d1(nout) = bad(trans(1))	      
 380	      d0(nout) = bad(trans(0))	      
 390	   CONTINUE
	ENDIF

*
* read "SKIP" records and initialize records read counter
	GOTO skip_in
  400	DO 425 scnt = 1,sf_skip(sfcnt)
  425	READ (lunit,END=9200,ERR=9000)
	GOTO 499
  450	DO 475 scnt = 1,sf_skip(sfcnt)
  475	READ (lunit,'(1X)',END=9200,ERR=9000)
  499	nout = 0		! is number of values (per var) read so far
	ncolmn = columns
	IF ( stream ) ncolmn = vpos
*
* Read depending on format, nrec counts number of records read
	GOTO read_in
* ... binary data
* Note: neither IOSTAT= nor ERR= branch is effective on binary read of ASCII
*	file ???? (tested on OSF only)
  500	READ (lunit,END=800,ERR=9500) (buff(rcnt),rcnt=1,ncolmn)
* if statement commented out 3/22/95 *kob*
*	IF ( ierr .NE. 0 ) GOTO 9500
	GOTO 690
* ... fixed format data
  600	READ (lunit,in_format,END=800,ERR=9500) (buff(rcnt),rcnt=1,ncolmn)
	GOTO 690
* ... free format data
  610	READ (lunit,*,END=800,ERR=9500) (buff(rcnt),rcnt=1,ncolmn)
	GOTO 690
* ... STREAM format data
*       Normally there will be only a single variable read so vpos=1 but
*       we are allowing multiple variables spaced apart in the file by
*       /COLUMNS= words (stored in ds_ntuple(dset)) 
* Note: neither IOSTAT= nor ERR= branch is effective on read past EOF ????
*	(tested on OSF only) (ok after all? - branch to 9500 no good *sh* 9/95)
  620	DO 622 vcnt = 1, vpos
  622	READ (lunit,rec=irec+(vcnt-1)*ds_ntuple(dset),ERR=9600) buff(vcnt)
	irec = irec + 1
!	GOTO 690

* Repeat the transfer operation if number of columns is a multiple of # vars
 690	DO 795 icol = 0, ncolmn-vpos, vpos
	   IF ( permuted ) THEN
	      nout = EZ_PERMUTED_INDEX()    ! increment permuted fashion
	   ELSE
	      nout = nout + 1
	   ENDIF
	   GOTO transloc
*
* Transfer to variable arrays depending on how many variables
 7010     d19(nout) = buff(trans(19)+icol)
 7110     d18(nout) = buff(trans(18)+icol)
 7210     d17(nout) = buff(trans(17)+icol)
 7310     d16(nout) = buff(trans(16)+icol)
 7410     d15(nout) = buff(trans(15)+icol)
 7510     d14(nout) = buff(trans(14)+icol)
 7610     d13(nout) = buff(trans(13)+icol)
 7710     d12(nout) = buff(trans(12)+icol)
 7810     d11(nout) = buff(trans(11)+icol)
 7910     d10(nout) = buff(trans(10)+icol)
  700      d9(nout) = buff(trans(9)+icol)
  710      d8(nout) = buff(trans(8)+icol)
  720      d7(nout) = buff(trans(7)+icol)
  730      d6(nout) = buff(trans(6)+icol)
  740      d5(nout) = buff(trans(5)+icol)
  750      d4(nout) = buff(trans(4)+icol)
  760      d3(nout) = buff(trans(3)+icol)
  770      d2(nout) = buff(trans(2)+icol)
  780      d1(nout) = buff(trans(1)+icol)
  790      d0(nout) = buff(trans(0)+icol)

 795	CONTINUE

  799   IF (nout .GE. nmax) GOTO 800
	GOTO read_in
*
* Finished, store records read and then exit
  800	nread = nout / ncol
	status = merr_ok
	GOTO 9950
*
* Error message
 9000	CALL TM_ERRMSG( merr_erreq, status, 'EZ_READ',
     .			dset, sfcnt, 'during SKIP reads',
     .			no_errstring, *9999                 )

 9100	CALL TM_ERRMSG( merr_notmounted, status, 'EZ_READ',
     .			dset, sfcnt, 'DATAFILE NOT FOUND',
     .			no_errstring, *9999                 )

 9200	CALL TM_ERRMSG( merr_eof, status, 'EZ_READ',
     .			dset, sfcnt,
     .			'during SKIP reads', no_errstring, *9950   )

 9300	CALL TM_ERRMSG( merr_dsetnf, status, 'EZ_READ',
     .			no_descfile, no_stepfile,
     .			no_errstring, no_errstring, *9999)

 9400	CALL TM_ERRMSG( merr_stepnf, status, 'EZ_READ',
     .			dset, no_stepfile,
     .			no_errstring, no_errstring, *9999)

 9500	BACKSPACE lunit
	READ (lunit,'(A)') errbuff
	CALL TM_ERRMSG( merr_erreq, status, 'EZ_READ',
     .			dset, sfcnt,
     .		'Last or next-to-last record read:'//cr//lf//errbuff,
     .			no_errstring, *9999                 )

 9600	CALL TM_ERRMSG( merr_erreq, status, 'EZ_READ',
     .			dset, no_stepfile,
     .			'error reading STREAM file', no_errstring, *9999)
*
* Close file and free lunit
 9950	CLOSE (lunit, ERR=9990)
 9990	CALL LIB_FREE_LUN (lunit)
        sf_lunit(sfcnt) = file_not_open    ! *sh* added 5/91
 9999	RETURN
	END
