	SUBROUTINE TM_ERRMSG ( 	status_in, status_out, routine,
     .				 dset_num, sf_num, 
     .				 expected, found, * )

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
* display explanatory message of error and calling routine, transfers error
* code to output variable, and always take alternate return to serve as error
* branch in calling program
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/19/85
* revision 1.00 - 04/01/87 - changes to calling statement to allow more
*			     complete but concise messages. (MAV)
* revision 1.10 - 06/01/87 - added "calling routine" parameter to call
* revision 1.11 - 09/24/88 - printing of dset, sf based on flags only. Flags
*			     now set to unspecified_int4 to facilitate this
* Unix/RISC port - 3/12/91 - Unix system error messages via UNIX CALL GERROR
*                          - TMAP_diag_on variable controls subroutine name out
*                          - changed error message format to "**TMAP ERR"
* added netCDF errors - 2/92 *sh*
* all output to be directed through TM_SPLIT_MESSAGE for optional GUI
* Linux Port - *kob* 5/97 - added call for linux_perror, which is a c 
*			    routine, because the appropriate routine 
*			    didn't exist in nag F90
*			  - f90 doesnt accept "D" in col 1 as a debug thing
*			    so ifdef around it
* V500 - *kob* 7/99 - update error codes to handle netCDF V3.4 differences
*
* arguments:
*	status_in	- error code from calling program
*	status_out	- error code (echoed) back to calling program
*	routine		- calling routine or program
*	dset_num	- data set number
*	sf_num		- step_file number
*	expected	- expected value, except in case where a limit is
*			  exceeded, in which case this is the limit. Or if
*			  special error, this might have another value.
*	found		- the value found when "expected" value was expected

#ifdef unix
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "gt_lib.parm"
	include 'xio.cmn_text'
	external xio_data
	include 'xerror_text.cmn_text'
	external xerror_text_data
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xstep_files.cmn_text'
	external xstep_files_data
	include 'xdiag_ctrl.cmn_text'
	external xdiag_ctrl_data
	include 'xcdferror.cmn_text'
	external xcdferror_data
!	include 'xalt_messages.cmn_text'	! w/ alt_messages
	external xalt_messages_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
	INCLUDE 'TMAP_FORMAT:XERROR_TEXT.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XDIAG_CTRL.CMN'
!	INCLUDE 'TMAP_FORMAT:XALT_MESSAGES.CMN'		! w/ alt_messages

	INTEGER		fnum, rmssts, rmsstv, iunit, condval, msgvec(8)
#endif

* argument definitions
	CHARACTER*(*)	routine, expected, found
	INTEGER		status_in, status_out, dset_num, sf_num

* local definitions
	CHARACTER*132	TM_CMPRSS, expbuff, fndbuff
	CHARACTER*256	errbuf	! introduced for GUI-SPLIT messages
	INTEGER		sf_len, des_len, TM_LENSTR1, msg_len, exp_len,
     .			fnd_len, rtn_len, cdferr
        INTEGER         cdfid, varid  ! cdf error bits

* initialize
	status_out = status_in	! transfer error code to output
	errbuf = ' '

* print calling routine line to identify where error occured
* (diagnostic, only - *sh* 3/15/91)
* add ifdef check for f90 - no D on col 1 *kob*
#if defined unix  && ! defined FORTRAN_90   
D        TMAP_diag_on = .TRUE.     ! compile if "-d_lines"-----Emacs: 
#endif
        IF ( TMAP_diag_on ) THEN
	   rtn_len = TM_LENSTR1(routine)
	   write (lunit_errors, *) ' ***** ERROR IN ',routine(:rtn_len),
     .				   ' *****'
        ENDIF

* Handle special messages first:

* error message from ERR= branch ?
* fetch the system error codes for most recent error

	IF ( status_in .EQ. merr_erreq ) THEN
#ifdef unix
#ifdef F90_SYSTEM_ERROR_CALLS
	  call linux_perror(expbuff)
#else
          CALL GERROR(expbuff)       ! get system error message
#endif
	  msg_len = TM_LENSTR1(expbuff)
	  WRITE (errbuf,1000) expbuff(:msg_len)
	  CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
#else
* VMS code by Mark Renton
	  CALL ERRSNS ( fnum, rmssts, rmsstv, iunit, condval )
	  IF ( ( rmssts .EQ. 0 ) .AND. ( rmsstv .EQ. 0 ) ) THEN
	    msgvec ( 1 ) = 1
	    msgvec ( 2 ) = condval
	    msgvec ( 3 ) = 0
	  ELSE
	    msgvec ( 1 ) = 1
	    msgvec ( 2 ) = rmssts
	    IF ( rmsstv .ne. 0 ) THEN
	      msgvec ( 1 ) = 2
	      msgvec ( 3 ) = rmsstv
	    ENDIF
	  ENDIF
* output the message but, SYS_PUTMSG needs a <cr> sent afterword
	  CALL SYS_PUTMSG ( msgvec, , , )
	  WRITE (errbuf,1000) expbuff(:msg_len)
	  CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
#endif

* Output the message for a regular error
	ELSE IF ( (status_in .GT. pregular) .AND.
     .		  (status_in .LE. pregular + num_regular + 1) ) THEN
	  msg_len = TM_LENSTR1(msg_regular(status_in - pregular))
	  WRITE (errbuf,1000)
     .		 msg_regular(status_in - pregular)(:msg_len)
	  CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
1000	  FORMAT(' **TMAP ERR: ',A)

* output a netCDF error (note: CDF uses error code -1)
* *kob* 7/99 - Update for netCDF v3.4
	ELSEIF ( (status_in .GE. pcdferr-100) .AND.
     .		 (status_in .LE. pcdferr + pcdferrmax) ) THEN
* xxxxx can we get filename from netCDF call ???           
* xxxxx cdfid and varid are available as dset_num, sf_num
           cdferr = status_in - pcdferr
           IF ( cdferr.EQ.-33 ) THEN
              WRITE (errbuf,1008) 'NCSYSERR'
           ELSEIF ( cdferr.EQ.0 ) THEN
              WRITE (errbuf,1008) 'No error'
           ELSEIF ( cdferr.GE. -66 .AND. cdferr.LE. -33) THEN
              WRITE (errbuf,1008) cdferrmsg(cdferr)
           ELSEIF ( cdferr .EQ. pcdferrmax ) THEN
              WRITE (errbuf, * ) 'error in netCDF file'
           ELSE
              WRITE (errbuf,1010) cdferr
           ENDIF
	   CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
 1008      FORMAT(' ** netCDF error: ', A)
 1010      FORMAT(' ** unknown netCDF error code:', I4)
           IF (expected .NE. no_errstring) THEN
	     errbuf = ' '
	     WRITE (errbuf,*) expected
	     CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
	   ENDIF
           IF (found .NE. no_errstring) THEN
	     errbuf = ' '
	     WRITE (errbuf,*) found
	     CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
	   ENDIF
           GOTO 4000   ! bypass normal error processing
* unrecognized error code ?
	ELSE
	  WRITE (errbuf,1100) status_in
	  CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
1100	  FORMAT(' *** Unknown error code:',I10,' *** ')
	END IF

* Output any info in expected and found strings (if any present)
	errbuf = ' '
	IF (expected .NE. no_errstring) THEN
	  expbuff = TM_CMPRSS(expected)
	  exp_len = TM_LENSTR1(expbuff)
	  IF (found .NE. no_errstring) THEN
	    fndbuff = TM_CMPRSS(found)
	    fnd_len = TM_LENSTR1(fndbuff)
	    WRITE (errbuf,1200) expbuff(1:exp_len),
     .				      fndbuff(1:fnd_len)
	    CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
1200	    FORMAT(13X,'Expected: ',A,' Found: ',A)
	  ELSE
	    WRITE (errbuf,1300) expbuff(1:exp_len)
	    CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
 1300       FORMAT(13X,A)
	  ENDIF
	ENDIF

* output step file and descriptor file if asked for
	IF ( dset_num .NE. no_descfile      .AND.
     .	     dset_num .NE. unspecified_int4       ) THEN
	  errbuf = ' '
	  des_len = TM_LENSTR1(ds_des_name(dset_num))
	  WRITE (errbuf,1300)
     .		 'Data set: '//ds_des_name(dset_num)(:des_len)
	  CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
	ENDIF

	IF ( sf_num .NE. no_stepfile      .AND.
     .	     sf_num .NE. unspecified_int4       ) THEN
	  errbuf = ' '
	  sf_len = TM_LENSTR1(sf_name(sf_num))
	  WRITE (errbuf,1300)
     .           'Data file: '//sf_name(sf_num)(:sf_len)
	  CALL TM_SPLIT_MESSAGE(lunit_errors, errbuf)
	ENDIF

* print blank line to set off error messages
! 4000   IF ( .NOT.alt_messages) WRITE (lunit_errors, *)
 4000	CONTINUE

* take alternate RETURN always
	RETURN 1

	END
