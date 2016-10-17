	SUBROUTINE DIAGNOSTIC_OUT( action, isub, pointer_type )

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
* output a line describing the activity in progress

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/5/87
* V200:  5/19/89 - re-ordered lo/hi_ss args
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420: 12/20/95 - do not print isub - changes too much between revisions
* Linux Port 12/96 *kob* - missing comma in 3000 format statement 
*                          discovered by f90 compiler
* 			 - Added a preprocessor include for tmap_dset.parm
*			   because it needed a preprocessor.
* V450: 7/7/97 - make a readable short name from VAR_CODE
* V500 *kob* 3/99- up VAR_CODE  to 64 chars and change buff24 to buff64
* V530 9/00 *sh* - output context or mr pointer rather than data set
* V552 *acm* 4/03- up VAR_CODE  to 128 chars and change buff64 to buff128
* V63  *acm* 10/09 Changes for gfortran build
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V680  5/12 *acm* Add dataset ID to diagnostic output
* V6.8  acm  6/12 Implement mode_6d_lab, set by default. If cancelled, the 
*                 output will not list info for E,F dimensions.
* V7    acm  6/16 Ticket 2437: Deprecating mode_6d_lab. 
*		  Adjust output to 6-D or 4-D grids

* calling argument declarations:
	INTEGER		isub, pointer_type
	CHARACTER*(*)	action

* internal argument declarations:
	LOGICAL	    FOUR_D_GRID
	INTEGER	    TM_LENSTR1, lenbuff, i, listdims
	CHARACTER   VAR_CODE*128, buff128*128, buff8*8

	include	'tmap_dims.parm'
	include	'ferret.parm'
#	include	"tmap_dset.parm"
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include 'xrisc.cmn'		! 12/94 SPLIT_LIST buffer


* If mode 6d_lab has been cancelled, we will list info only for 
* the first 4 dimensions.  This test fails for some EX# expressions.

	listdims = nferdims
	IF ( pointer_type .EQ. point_to_cx ) THEN
	   IF ( FOUR_D_GRID(cx_grid(isub)) ) listdims = 4
	ELSe
	   IF ( FOUR_D_GRID(mr_grid(isub)) ) listdims = 4
	ENDIF

	risc_buff = ' '

* readable variable name code
	IF ( pointer_type .EQ. point_to_cx ) THEN
 	   buff128 = VAR_CODE( cx_category(isub), cx_variable(isub) )
	ELSE
	   buff128 = VAR_CODE( mr_category(isub), mr_variable(isub) )
	ENDIF
	IF ( buff128(:2) .EQ. '(C' ) THEN
! ... use low order 2 digits from 3-digit numbers (shortening to 2-digits)
	   buff8 = '(C'//buff128(4:5)//',V'//buff128(9:10)
	ELSE
	   lenbuff = TM_LENSTR1(buff128)
! ... assemble short name from beginning and end of full name 
	   IF (lenbuff .GT. 8 ) THEN
	      buff8 = buff128(1:3)//'-'//buff128(lenbuff-3:lenbuff)
	      IF (buff128(1:1) .EQ. '{') buff8 = buff128(1:4)//'...'//'}'
	   ELSE
	      buff8 = buff128
	   ENDIF
	ENDIF

      IF (listdims .EQ. nferdims) THEN

	IF ( pointer_type .EQ. point_to_cx ) THEN
	   WRITE ( risc_buff,3000 )
     .			action,
     .		 	buff8,
     .			'C',isub,
     .			cx_data_set(isub),
     .			(cx_lo_ss(isub,i),cx_hi_ss(isub,i),i=1,listdims)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )
	ELSE
	   WRITE ( risc_buff,3000 )
     .			action,
     .			buff8,
     .			'M',isub,
     .			mr_data_set(isub),
     .			(mr_lo_ss(isub,i),mr_hi_ss(isub,i),i=1,listdims)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )
	ENDIF

      ELSE

	IF ( pointer_type .EQ. point_to_cx ) THEN
	   WRITE ( risc_buff,3100 )
     .			action,
     .		 	buff8,
     .			'C',isub,
     .			cx_data_set(isub),
     .			(cx_lo_ss(isub,i),cx_hi_ss(isub,i),i=1,listdims)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )
	ELSE
	   WRITE ( risc_buff,3100 )
     .			action,
     .			buff8,
     .			'M',isub,
     .			mr_data_set(isub),
     .			(mr_lo_ss(isub,i),mr_hi_ss(isub,i),i=1,listdims)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )
	ENDIF

      ENDIF

 3000	FORMAT (1X,A,1X,A8,1X,A1,':',I3,' dset:',i4,
     . ' I:',2I5,'  J:',2I5,'  K:',2I5,'  L:',2I5,'  M:',2I5,'  N:',2I5)
 3100	FORMAT (1X,A,1X,A8,1X,A1,':',I3,' dset:',i4,
     . ' I:',2I5,'  J:',2I5,'  K:',2I5,'  L:',2I5)

	RETURN
	END	
