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
* V500 3/99 *kob* - Initialize buff at 610 to prevent use of previous values
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm* 10/01  remove VMS include statements
*
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use character constant instead
* v604 *acm* 7/07 increase length of ds_name, ds_des_name
* V68  *acm* 1/12 The variables should be declared REAL (conversion to double prec.)
* V694 11/14 *acm* Ticket 2217: permuted ASCII reading in 6D
* V701  7/16 *kms* Compare ds_name to char_init2048 instead of char_init1024
* 10/2022 *acm* Code cleanup: Rewrite to avoid ASSIGN statements and computed GOTO

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
	REAL		d0(ncol*nread), d1(ncol*nread), d2(ncol*nread),
     .			d3(ncol*nread), d4(ncol*nread), d5(ncol*nread), 
     .			d6(ncol*nread), d7(ncol*nread), d8(ncol*nread), 
     .			d9(ncol*nread),
     .			d10(ncol*nread),d11(ncol*nread),d12(ncol*nread),
     .                  d13(ncol*nread),d14(ncol*nread),d15(ncol*nread),
     .                  d16(ncol*nread),d17(ncol*nread),d18(ncol*nread),
     .                  d19(ncol*nread)
        REAL buff(columns)

*
* Include files
	include 'ez_lib.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xez_info.cmn_text'
	external xez_info_data
	include 'xez_order.cmn_text'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xstep_files.cmn_text'
	external xstep_files_data
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
	INTEGER*4	icol, nout, ncolmn
	INTEGER*4	itemp, i, irec, ierr
	INTEGER*4	trans(0:maxezvars-1)
	DATA 		(trans(i),i=0,maxezvars-1) /maxezvars*-1/
	LOGICAL		unformatted, file_exists, file_open,
     .			permuted, stream
	REAL		bad(0:maxezvars-1)



* Does data set exist?
	IF (ds_name(dset) .EQ. char_init2048) GOTO 9300
*
* Locate Step file entry
	DO sfcnt = 1,maxstepfiles
	  IF (sf_setnum(sfcnt) .EQ. dset) GOTO 20
	ENDDO
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
	      bad(tpos) = ds_bad_flag(vcnt)
	    ENDIF
	  ENDIF
  100	CONTINUE

* is the input permuted?

	permuted = .FALSE.
	DO 110 i = 1, nferdims
	   IF ( ds_ordering(i,vfirst) .NE. i ) permuted = .TRUE.
 110	CONTINUE
!	permuted = .TRUE.   ! debug statement
	IF ( permuted ) CALL EZ_INIT_PERMUTE( vfirst )


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

c	print *, 'ez_read file_format, permuted ',  file_format, ' ', permuted 

*
* open the file
	IF ( stream ) THEN       ! 5/94
	   OPEN	(UNIT = lunit,
     .		 FILE = ds_des_name(dset),
     .		 FORM = 'UNFORMATTED',
     .		 ACCESS = 'DIRECT',
     .		 RECL = 4,
     .		 STATUS = 'OLD',
     .		 ERR = 9000)
	ELSE
	   OPEN	(UNIT = lunit,
     .		 FILE = ds_des_name(dset),
     .		 FORM = file_format,
     .		 ACCESS = 'SEQUENTIAL',
     .		 STATUS = 'OLD',
     .		 ERR = 9000)
	ENDIF
	sf_lunit(sfcnt) = lunit    ! *sh* dset -> sfcnt 5/91
*
* if input is to be permuted then it is essential to pre-fill the arrays
* that will receive the data - this because a fragmentary read (less data in
* the file then the grid dimensions would imply ) will end
* at a point in the middle of the output arrays.
	IF ( permuted ) THEN
c	   print *, 'ez_read: tpos, nmax', tpos, nmax 
	   DO nout = 1, nmax
	      IF (tpos .GE. 20) d19(nout) = bad(trans(19))	      
	      IF (tpos .GE. 19) d18(nout) = bad(trans(18))	      
	      IF (tpos .GE. 18) d17(nout) = bad(trans(17))	      
	      IF (tpos .GE. 17) d16(nout) = bad(trans(16))	      
	      IF (tpos .GE. 16) d15(nout) = bad(trans(15))	      
	      IF (tpos .GE. 15) d14(nout) = bad(trans(14))	      
	      IF (tpos .GE. 14) d13(nout) = bad(trans(13))	      
	      IF (tpos .GE. 13) d12(nout) = bad(trans(12))	      
	      IF (tpos .GE. 12) d11(nout) = bad(trans(11))	      
	      IF (tpos .GE. 11) d10(nout) = bad(trans(10))	      
	      IF (tpos .GE. 10)  d9(nout) = bad(trans(9))	      
	      IF (tpos .GE.  9)  d8(nout) = bad(trans(8))	      
	      IF (tpos .GE.  8)  d7(nout) = bad(trans(7))	      
	      IF (tpos .GE.  7)  d6(nout) = bad(trans(6))	      
	      IF (tpos .GE.  6)  d5(nout) = bad(trans(5))	      
	      IF (tpos .GE.  5)  d4(nout) = bad(trans(4))	      
	      IF (tpos .GE.  4)  d3(nout) = bad(trans(3))	      
	      IF (tpos .GE.  3)  d2(nout) = bad(trans(2))	      
	      IF (tpos .GE.  2)  d1(nout) = bad(trans(1))	      
	      IF (tpos .GE.  1)  d0(nout) = bad(trans(0))	      
	   ENDDO
	ENDIF

*
* read "SKIP" records and initialize records read counter

	IF ( unformatted ) THEN

	   DO scnt = 1,sf_skip(sfcnt)
	      READ (lunit,END=9200,ERR=9000)
	   ENDDO 

	ELSEIF ( in_format(:4) .EQ. 'FREE' ) THEN

	   DO scnt = 1,sf_skip(sfcnt)
	      READ (lunit,'(1X)',END=9200,ERR=9000)
  	   ENDDO

	ELSEIF ( stream ) THEN

	  irec = sf_skip(sfcnt) + 1     ! /SKIP=words

	ELSE

	   DO scnt = 1,sf_skip(sfcnt)
	      READ (lunit,'(1X)',END=9200,ERR=9000)
  	   ENDDO

	ENDIF

	nout = 0		! is number of values (per var) read so far
	ncolmn = columns
	IF ( stream ) ncolmn = vpos

* Read depending on format, nrec counts number of records read
	DO WHILE (nout .lt. nmax)


*   set up ASSIGN statements (for GOTO variable statements)
	IF ( unformatted ) THEN

	  READ (lunit,END=800,ERR=9500) (buff(rcnt),rcnt=1,ncolmn)

	ELSEIF ( in_format(:4) .EQ. 'FREE' ) THEN
	  
           DO rcnt = 1,ncolmn
	      buff(rcnt) = -1.E34
           ENDDO
	   READ (lunit,*,END=800,ERR=9500) (buff(rcnt),rcnt=1,ncolmn)

	ELSEIF ( stream ) THEN			! "STREAM" direct access binary

* ... STREAM format data
*       Normally there will be only a single variable read so vpos=1 but
*       we are allowing multiple variables spaced apart in the file by
*       /COLUMNS= words (stored in ds_ntuple(dset)) 
* Note: neither IOSTAT= nor ERR= branch is effective on read past EOF ????
*	(tested on OSF only) (ok after all? - branch to 9500 no good *sh* 9/95)

	   DO vcnt = 1, vpos
	      READ (lunit,rec=irec+(vcnt-1)*ds_ntuple(dset),ERR=9600) buff(vcnt)
	   ENDDO
	   irec = irec + 1

	ELSE

	   READ (lunit,in_format,END=800,ERR=9500) (buff(rcnt),rcnt=1,ncolmn)

	ENDIF

* Repeat the transfer operation if number of columns is a multiple of # vars
	DO icol = 0, ncolmn-vpos, vpos
	   IF ( permuted ) THEN
	      nout = EZ_PERMUTED_INDEX()    ! increment permuted fashion
	   ELSE
	      nout = nout + 1
	   ENDIF
*
* Transfer to variable arrays depending on how many variables

   	   IF (tpos .GE. 20) d19(nout) = buff(trans(19)+icol)
   	   IF (tpos .GE. 19) d18(nout) = buff(trans(18)+icol)
   	   IF (tpos .GE. 18) d17(nout) = buff(trans(17)+icol)
   	   IF (tpos .GE. 17) d16(nout) = buff(trans(16)+icol)
   	   IF (tpos .GE. 16) d15(nout) = buff(trans(15)+icol)
   	   IF (tpos .GE. 15) d14(nout) = buff(trans(14)+icol)
   	   IF (tpos .GE. 14) d13(nout) = buff(trans(13)+icol)
   	   IF (tpos .GE. 13) d12(nout) = buff(trans(12)+icol)
   	   IF (tpos .GE. 12) d11(nout) = buff(trans(11)+icol)
   	   IF (tpos .GE. 11) d10(nout) = buff(trans(10)+icol)
   	   IF (tpos .GE. 10) d9(nout) = buff(trans(9)+icol)
   	   IF (tpos .GE.  9) d8(nout) = buff(trans(8)+icol)
   	   IF (tpos .GE.  8) d7(nout) = buff(trans(7)+icol)
   	   IF (tpos .GE.  7) d6(nout) = buff(trans(6)+icol)
   	   IF (tpos .GE.  6) d5(nout) = buff(trans(5)+icol)
   	   IF (tpos .GE.  5) d4(nout) = buff(trans(4)+icol)
   	   IF (tpos .GE.  4) d3(nout) = buff(trans(3)+icol)
   	   IF (tpos .GE.  3) d2(nout) = buff(trans(2)+icol)
   	   IF (tpos .GE.  2) d1(nout) = buff(trans(1)+icol)
	   IF (tpos .GE.  1) d0(nout) = buff(trans(0)+icol)
	ENDDO

	ENDDO		! READ loop: DO WHILE (nout .lt. nmax)

  800	CONTINUE	! hit an END on read
*
* Finished, store records read and then exit
	nread = nout / ncol
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
