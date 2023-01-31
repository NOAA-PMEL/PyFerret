	SUBROUTINE EZ_COUNT_DSET ( dset, ezform, nread, status  )
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
* Ansley Manke 7/2014
* 
* Subroutine opens an EZ data set (ASCII and unformatted non_TMAP data files) 
* and returns a count count the records in the datset, minus the number of 
* records skipped. Addressing ticket 2182.
*
* V701  7/16 *kms* Compare ds_name to char_init2048 instead of char_init1024
* 10/2022 *acm* Code cleanup: Rewrite to avoid ASSIGN statements

* Arguments:
*	dset	- data set number
*	ezform	- file format
*	nread	- (output) number of data records in file
*
	INTEGER*4	dset, nread, status
	CHARACTER*(*)	ezform

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
	CHARACTER*80	errbuff
	CHARACTER*99	in_format
	INTEGER*4	sfcnt, lunit, scnt, rcnt
	INTEGER*4	skip_in, read_in, nout
	INTEGER*4	itemp, i, irec, ierr
	LOGICAL		unformatted, file_exists, file_open, stream
        REAL		buff

* Does data set exist?
	IF (ds_name(dset) .EQ. char_init2048) GOTO 9300
*
* Locate Step file entry
	DO 10 sfcnt = 1,maxstepfiles
	  IF (sf_setnum(sfcnt) .EQ. dset) GOTO 20
  10	CONTINUE
	GOTO 9400

* Initialize
 20	nread = 0

*   get file format
	stream = ezform(:3) .EQ. 'STR'
	unformatted = ezform(:3) .EQ. 'UNF'
	IF ( unformatted ) THEN
	  file_format = 'UNFORMATTED'
	ELSE
	  file_format = 'FORMATTED'
	  in_format = ezform		! Run time format info
	ENDIF

* Check file validity if necessary
  160	IF (.NOT. sf_valid(sfcnt)) THEN
!	  INQUIRE (	FILE = sf_name(sfcnt),	! sf_name is too short 10/89
	  INQUIRE (	FILE = ds_des_name(dset),
     .			RECL = itemp,	! temporary var needed by DEC/RISC
     .			EXIST = file_exists,
     .			OPENED = file_open,
     .			ERR = 9000)
	  IF (.NOT. file_exists) GOTO 9100
* calculate RECL in longwords for unformatted (if not open)             
          IF (.NOT. file_open .AND. unformatted )
     .	                         sf_reclen(sfcnt) = sf_reclen(sfcnt)/4
        ENDIF
*
* get free VMS logical unit #   
	CALL LIB_GET_LUN (lunit)
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


	ELSEIF ( stream ) THEN			! "STREAM" direct access binary
	   irec = sf_skip(sfcnt) + 1     ! /SKIP=words
  	   nout = 0		! is number of values (per var) read so far


	ELSE

  	   DO scnt = 1,sf_skip(sfcnt)
  	      READ (lunit,'(1X)',END=9200,ERR=9000)
  	   ENDDO

  	ENDIF 

	nout = 0

* Read depending on format, nrec counts number of records read

	DO WHILE (.TRUE.)  ! read until END or ERR
	
* ... binary data

	   IF ( unformatted ) THEN
  	      READ (lunit,END=800,ERR=9500) 

* ... free format data
	   ELSEIF ( in_format(:4) .EQ. 'FREE' ) THEN
              READ (lunit,*,END=800,ERR=9500) 

* ... stream
	   ELSEIF ( stream ) THEN	
              READ (lunit,rec=irec,ERR=9600) buff
	      irec = irec + 1
   
* ... fixed format data
	   ELSE
  	     READ (lunit,*,END=800,ERR=9500) 
	   ENDIF

  	   nout = nout + 1

	ENDDO

*
* Finished, store records read and then exit
  800	nread = nout 
	status = merr_ok
	GOTO 9950
*
* Error message
 9000	CALL TM_ERRMSG( merr_erreq, status, 'EZ_COUNT',
     .			dset, sfcnt, 'during SKIP reads',
     .			no_errstring, *9999                 )

 9100	CALL TM_ERRMSG( merr_notmounted, status, 'EZ_COUNT',
     .			dset, sfcnt, 'DATAFILE NOT FOUND',
     .			no_errstring, *9999                 )

 9200	CALL TM_ERRMSG( merr_eof, status, 'EZ_COUNT',
     .			dset, sfcnt,
     .			'during SKIP reads', no_errstring, *9950   )

 9300	CALL TM_ERRMSG( merr_dsetnf, status, 'EZ_COUNT',
     .			no_descfile, no_stepfile,
     .			no_errstring, no_errstring, *9999)

 9400	CALL TM_ERRMSG( merr_stepnf, status, 'EZ_COUNT',
     .			dset, no_stepfile,
     .			no_errstring, no_errstring, *9999)

 9500	BACKSPACE lunit
	READ (lunit,'(A)') errbuff
	CALL TM_ERRMSG( merr_erreq, status, 'EZ_COUNT',
     .			dset, sfcnt,
     .		'Last or next-to-last record read:'//cr//lf//errbuff,
     .			no_errstring, *9999                 )

 9600	CALL TM_ERRMSG( merr_erreq, status, 'EZ_COUNT',
     .			dset, no_stepfile,
     .			'error reading STREAM file', no_errstring, *9999)

*
* Close file and free lunit
 9950	CLOSE (lunit, ERR=9990)
 9990	CALL LIB_FREE_LUN (lunit)
        sf_lunit(sfcnt) = file_not_open    ! *sh* added 5/91
 9999	RETURN
	END
