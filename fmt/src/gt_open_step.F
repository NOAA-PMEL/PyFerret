	SUBROUTINE GT_OPEN_STEP ( sf_num,status )

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
* Open a step file. If the validity of the file hasn't been checked yet
* check its firt 4 bytes as an indication of the right file
* type, and make sure it's first and last time step values are known,
* if the file doesn't exist return status but DONT call ERRMSG

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/15/85
* revision 1.00 - 08/26/86 - reflects changes to descriptor,set and data
*			     file formats (mark verschell)
* revision 1.10 - 11/19/86 - changed handling of error conditions (mav)
* revision 1.20 - 09/19/88 - Uses EXIST paramater to check if file on line
*			     Uses OPEN to see if should getting longwords or
*			     bytes for RECLEN
* revision 1.30 - 02/15/89 - Added call to TM_ONLY1_STEP
* revision 1.31 - 1/91 *sh* syntax changes to accomodate DEC RISC compiler
* Unix/RISC port - 3/14/91 *sh*: file record length computed and stored in
*                  sf_reclen during gt_init_dset
*                - flag binary conversion needs in sf_cpu_type 
*                - 4/24 - uses TM_INQ_PATH and tmap_tsteps variable
* 6/91 ------------> kob - Time Independent data capability added  
*                - included file tmap_format/gt_lib.parm
*                    needed for declaration of unspecified_val8
* 8/91 bug fix   - *kob*
*                - new stepfile info was being stored for old
*                  stepfile.  Created prob's when using datasets
*                  written on different machines.  Moved storing of new
*                  stepfile info before status ok exit. (at 1000)
* SUN port - 1/30/92 -kob- 1) Moved 'external xmachine_data out of ifdef so it 
*                              will become valid on the vax.
*                          2) xmachine_byte.cmn on SUN and VAX
*                             and xmachine_int1.cmn for DECstation because
*                             because it understands INTEGER*1
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
*
* IBM port 14apr94 *kob*  : IBM would not allow readonly in file opens, so an 
*			   "#ifdef READONLY_ALLOWED" was placed around the readonly
*			    in open statement.
* arguments:
*	sf_num - "step_file" number (in COMMON/XSTEP_FILE/) of file to open
*	status	- (output) error flag 

#ifdef unix
	include 'tmap_format/tmap_errors.parm'	! error codes
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/gt_lib.parm'        ! added 6/91 --->kob
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xio.cmn_text'
	external xio_data
	include 'tmap_format/xsys_names.cmn_text'
	external xsys_names_data
* xmacine_byte.cmn on sun becauses it uses BYTE where ULTRIX uses INTEGER*1
#ifdef sun
      include 'tmap_format/xmachine_byte.cmn'
#else
      include 'tmap_format/xmachine_int1.cmn'
#endif

#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'	! error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
	INCLUDE 'TMAP_FORMAT:XSYS_NAMES.CMN'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif
* moved out of ifdef unix to become valid on vax as well 1/92 -kob-
	EXTERNAL XMACHINE_DATA

* internal variable declarations
        INTEGER*2       nvari
	INTEGER		sf_num,lunit,idummy,status,cptype
	CHARACTER*4	rec_code,nskip_char
	CHARACTER*13	TM_STRING
        CHARACTER*128   TM_INQ_PATH, tstep_file
        LOGICAL         TM_FPEQ
	REAL*8		time_step
#ifdef sun
        BYTE            author_cpu
#else
#	if unix  
        	INTEGER*1       author_cpu
#	else
        	BYTE            author_cpu
#	endif
#endif


* initialize: get file characteristics
* *sh* 3/91: removed RECL = sf_reclen(sf_num)
* *sh* 4/91: path/logical name (tmap_tsteps) not stored in descriptor file
	IF ( .NOT. sf_valid(sf_num) ) THEN
           tstep_file = TM_INQ_PATH( sf_name(sf_num), tmap_tsteps, ' ',
     .                          .TRUE., status )
           IF ( status .NE. merr_ok ) GOTO 9010
           sf_name(sf_num) = tstep_file     ! save exact path
        ENDIF

* get free logical unit #
	CALL LIB_GET_LUN (lunit)

	OPEN   (UNIT = lunit,
     .		FILE = sf_name(sf_num),
     .		RECL = sf_reclen(sf_num),
     .		FORM = 'UNFORMATTED',
     .		ACCESS = 'DIRECT',
     .		STATUS = 'OLD',
#ifdef READONLY_ALLOWED
     .		READONLY,
#endif
     .		ERR = 5000)

* we're done if validity was already checked
	IF (sf_valid(sf_num)) GOTO 1000

* ----------- check validity of step file --------
* check to see if the first records should be skipped
	READ (UNIT=lunit, REC=1, ERR=5000) rec_code,nskip_char
	IF (rec_code .EQ. 'SKIP') THEN
	  READ (nskip_char, FMT='(I4)', ERR=5000) sf_skip(sf_num)
	ELSE
	  sf_skip(sf_num) = 0
	ENDIF

* check validity of first time step
	READ (UNIT=lunit, REC=sf_skip(sf_num)+1, ERR=5000)
     .       rec_code, time_step, idummy, nvari, author_cpu
	IF (rec_code .NE. 'STEP') CALL TM_ERRMSG
     .		(merr_stepform, status, 'GT_OPEN_STEP',
     .		 sf_setnum(sf_num), sf_num,
     .		 'STEP'//' TSTEP='//TM_STRING(sf_1step(sf_num)),
     .		 rec_code//' TSTEP='//TM_STRING(time_step), *9000)

* file written on a different machine ? convert machine words
	IF ( author_cpu .NE. active_cpu ) THEN
	   cptype = author_cpu
	   CALL TM_INT2_SWAP( nvari, cptype )
	   CALL TM_DFP_CONVERT( time_step, cptype )
	ENDIF

* (changed to single precision compare (matching gt_find_rec) 3/91)
* modified if statment to look for unspecified_val8 in sf_1step,
*   indicating time-independent data set  ------------> kob 6/91
        IF ( .NOT. TM_FPEQ(SNGL(time_step), SNGL(sf_1step(sf_num))) 
     . .AND. (sf_1step(sf_num) .NE. unspecified_val8))
     .	CALL TM_ERRMSG (merr_stpmtch, status, 'GT_OPEN_STEP',
     .		 sf_setnum(sf_num), sf_num,
     .		 TM_STRING(sf_1step(sf_num)), TM_STRING(time_step), *9000)

* store step file info.  moved up from below 1000     !8/91
	sf_valid(sf_num)      = .TRUE.
        sf_cpu_type(sf_num)   = author_cpu
        sf_compress(sf_num)   = .FALSE.         ! not yet implemented

* successful completion - leave step_file open   
* and store only logical unit number             !8/91
1000	status = merr_ok
	sf_lunit(sf_num)      = lunit

* close any other open step_files for this data set
	CALL TM_ONLY1_STEP(sf_setnum(sf_num), sf_num, status)
	GOTO 9010

* error exit - close file
 5000	CALL TM_ERRMSG (merr_erreq, status, 'GT_OPEN_STEP',
     .		 	sf_setnum(sf_num), sf_num,
     .			no_errstring, no_errstring, *9000)
 9000	CLOSE (UNIT = lunit, ERR=9010)		! ERR= in case file wasn't open
 9010	RETURN
	END
