	SUBROUTINE IS_READ ( memory, *, *, status )

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
* read a grid of data from a system file
* this routine merely looks at the class of data set to decide on the
* appropriate underlying routine to actually get the data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/19/87
* revision 0.1 - 9/11/87 - changed check of ds_format to ds_type 
*			   and 'GT' --> '  GT' for new TMAP library
* revision 0.2 - 11/2/88 - IS_READ_GT --> IS_READ_TM (ss-symmetrical)
* V200: 10/25/89 - EZ data type added
* 	 2/27/90 - unspecified limits are set with FLESH_OUT_AXIS
* V230:  3/2/92 - added netCDF support
* V312: 5/94 - array "memory" as a calling argument
* V320: 2/95 - allow for category cat_file_rev (a file variable with
*		reversed axes)
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA statment rather than a
*	        PARAMETER statement because variable was modified in code
* 	      - Added a preprocessor include for tmap_dset.parm
*	        because it needed a preprocessor.
*  7/98 *js*  - Hacked to add support for permuted binary file reads
* V530: *sh* 11/00 - delimited file reading added

* calling argument declarations:
	INTEGER	status
	REAL	memory(*)
* normal exit is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

* internal variable declarations:
	LOGICAL		reprimand
	DATA       	reprimand/.TRUE./
	INTEGER 	CGRID_AXIS, cx, dset_num, idim, cat

	include	'tmap_dims.parm'
#	include	"tmap_dset.parm"
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'

* --- end of introductory code ---

* initialize
	cx = is_cx( isp )
	dset_num = cx_data_set( cx )

* for the duration of this routine replace cat_revfile_var with cat_file_var
* (since most routines cannot cope with cat_revfile_var)
	cat = cx_category(cx)
	cx_category(cx) = cat_file_var

* ensure that all needed limits are known
	DO 100 idim = 1, 4
	   IF ( cx_lo_ss( cx, idim ) .NE. unspecified_int4 ) GOTO 100
	   IF ( CGRID_AXIS( idim, cx ) .EQ. mnormal ) GOTO 100
	   IF ( reprimand ) THEN
	      CALL WARN
     .		('FERRET is more efficient if regions are fully specified' )
	      CALL WARN( ww_dim_name(idim)//
     .		' axis limits are unspecified - possibly others, too' )
	      reprimand = .FALSE.		! only remind them once
	   ENDIF
	   CALL FLESH_OUT_AXIS( idim, cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000		! should never happen
!	   CALL ERRMSG( ferr_limits, status,
!     .			ww_dim_name(idim)//' axis limits are unknown for '//
!     .			VAR_CODE( cat_file_var, cx_variable(cx) ), *5000 )
 100	CONTINUE

* choose the appropriate routine to read the data
	IF ( ds_type( dset_num ) .EQ. pds_type_ez ) THEN
* all stream formatted files now read with new fast binary reads
	   IF ( ds_parm_text(1,dset_num)(:3) .eq. 'STR') THEN
	      CALL READ_BINARY( memory, cx, is_mr(isp), status)
	   ELSEIF ( ds_parm_text(1,dset_num)(:4) .eq. 'DELI') THEN
	      CALL READ_DELIMITED( memory, cx, is_mr(isp), status)
	   ELSEIF ( ds_parm_text(1,dset_num)(:4) .eq. 'PERL') THEN
	      CALL READ_PERL( memory, cx, is_mr(isp), status)
	   ELSE
	      CALL READ_EZ( memory, cx, is_mr(isp), status )
	   ENDIF
	   IF ( status .EQ. ferr_ok ) GOTO 2
	ELSE
	   CALL READ_TM( memory, cx, is_mr(isp), status )
	   IF ( status .EQ. ferr_ok ) GOTO 2
	ENDIF

* error exit
 5000	cx_category(cx) = cat
 	RETURN

* never need components since TM IO library handles 4D accesses as of 11/88
! 1	cx_category(cx) = cat
!	RETURN 1

* done - we got what was asked for
 2	cx_category(cx) = cat
	RETURN 2
	END	
