	SUBROUTINE SHOW_REGION( cx )

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
* execute SHOW REGION for a single context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/23/87
* revision 0.1 - 3/23/88 - enlarged cx_DIM_STR to accomodate date/time strings
* V200:  5/19/89 - reordered lo/hi_ss args
* Unix/RISC port - 2/27/91 - mystery error with version of FORTRAN compiler
*		          eliminated using temporary buffer for cx_dim_str
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V550: *acm*10/02- Changed cx_name to char*24, so use TM_LENSTR1 with it.
*       *ACM*11/02- Did NOT need the above change with TM_LENSTR1.
* v554: *acm* 11/03 call SPLIT_LIST with open string length; region names now up to 24 long.

	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'

* calling argument declarations
	INTEGER	cx

* local parameter declarations:
	LOGICAL	    full_precision
	PARAMETER ( full_precision = .TRUE. )

* local variable declarations:
	INTEGER  idim, slen
	CHARACTER cx_DIM_STR*48

* write region name
	IF ( cx .EQ. cx_last ) THEN
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				'default region:',0)
	ELSE
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		'region '//cx_name(cx),0)
	ENDIF

* describe each axis - produce lines as complex as:"X=lo:hi@xxx:nn (transform)"
	DO 100 idim = 1, 4
* axis unspecified ?
	   IF (  cx_by_ss(idim,cx)
     .	        .AND. cx_lo_ss (cx,idim) .EQ. unspecified_int4
     .	   .OR..NOT.cx_by_ss(idim,cx)
     .		.AND.  cx_lo_ww (idim,cx) .EQ. unspecified_val8) THEN
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .				'        '//ww_dim_name(idim)//
     .				'/'//ss_dim_name(idim)//' is unspecified',
     .				26 )
	   ELSE

* write it out
	      CALL SPLIT_LIST(pttmode_explct, ttout_lun,'        '
     .		//CX_DIM_STR(idim,cx,':',full_precision,slen),0)

	   ENDIF

 100	CONTINUE	      

	RETURN
	END
