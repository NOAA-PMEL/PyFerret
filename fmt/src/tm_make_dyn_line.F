	SUBROUTINE TM_MAKE_DYN_LINE(idim, lo, hi, del,
     .					units, modulo,
     .					outline, status)

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
* Create a dynamic (always) line base on the input description
* To create a dynamic line that is based on another line use
* TM_GET_LIKE_DYN_LINE, instead
* Will create a line description based on the input arguments and
* compare it to other line definitions in common. Unlike
* tm_get_like_line (and unlike TM_GET_LIKE_DYN_GRID) this will check
* dynamic lines (only).
* If there is a match then "outline" returns with the number of the match.
* If there is no match, then a new dynamic line is created.
* "outline" returns with the number of the new dynamic line.

* created 1/98 *sh* - based on tm_get_like_dyn_line

* Arguments
	LOGICAL		modulo
	INTEGER		idim, outline, status
	REAL*8		lo, hi, del
	CHARACTER*(*)	units

* argument description:
* in	idim	  - axis orientation: 1-4
* in	units	  - units string for axis (max 16 characters)
*		    used when labeling the axis
*		    if creating a time axis format should be
*		    "units since dd-mmm-yyyy:hh:mm:ss" with 
*		    max of 16 characters for the unit string
* in	modulo	  - if creating a modulo axis this is TRUE (untested on 1/98)
* in	lo,hi,del - description of line (note REAL*8)
* out	outline	  - returned line

* INCLUDE FILES
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/implicit.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	include 'tmap_format/xunits.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:IMPLICIT.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
#endif
*
* Local definitions
	INTEGER		TM_FIND_LIKE_LINE, TM_FIND_LIKE_DYN_LINE,
     .			TM_LENSTR1, TM_UNIT_ID,
     .			inline, line_match, iunit, since_T0

	REAL*8 		dp_epsilon
	PARAMETER 	( dp_epsilon = 1.D-10)

* allocate a scratch line 
	CALL TM_ALLO_DYN_LINE( inline, status )
	IF ( status .NE. merr_ok ) RETURN

* initialize
	line_t0(inline)        = char_init20
	line_tunit(inline)     = 0

* basic defaults on the line to be created
	line_parent(inline)    = 0
	line_regular(inline)   = .TRUE.
	line_class(inline)     =  pline_class_basic
	line_direction(inline) = axis_orients(idim)

* set user-specified (boolean) modulo flag
	line_modulo(inline)    = modulo		! "TRUE" untested 1/98 !!!

* set user-specified units -- attempting to interpret them (a pain!)
	line_units(inline)     = units
	IF ( idim .EQ. t_dim ) THEN
* ... sloppy check for "since dd-mmm-yyyy" -- misses mixed case "Since"
	   since_T0 = MAX( INDEX(units,'since'), INDEX(units,'SINCE') )
	   IF ( since_T0 .GT. 2 ) THEN
	     line_units(inline) = units(:since_T0-1) ! just the "units" part 
	     IF ( TM_LENSTR1(units) .GT. since_T0 + 5 ) THEN
	       CALL TM_DATE_REFORMAT(units(since_T0+6:),pdate_numslashnum,
     .				     line_t0(inline), pdate_vax, status )
	       IF ( status .NE. merr_ok ) GOTO 9000
	     ENDIF
	   ENDIF
	ENDIF

* decode the units - check the validity
	line_unit_code(inline) = 0
	IF ( units .NE. ' ' ) THEN
           line_unit_code(inline) = TM_UNIT_ID( line_units(inline) )
	   iunit = line_unit_code(inline)	
           IF ( iunit .EQ. 0 ) THEN
	     CALL WARN('unknown axis units: '//units)
	   ELSEIF (iunit .EQ. -5) THEN    ! "-5" is "MONTHS"
               CALL WARN(units//
     .			' is ambiguous ... using 1/12 of 365 days.') 
           ELSEIF ( iunit .NE. 0 ) THEN
* ... Issue warning for inappropriate units like HOURS on X ax
*     and render the units unconvertible
             IF (  ( idim.EQ.t_dim .AND.
     .             ( iunit.LT.pun_1st_time.OR.iunit.GT.pun_last_time ) )
     .        .OR. ( idim.NE.t_dim .AND.
     .             ( iunit.LT.pun_1st_len .OR.iunit.GT.pun_last_len  ) )
     .        .OR. ( idim.EQ.z_dim .AND.
     .               iunit.EQ.pun_degrees )  ) THEN     ! 7/96
               CALL WARN
     .               (units//' appear to conflict with orientation:'
     .                  //' no conversions possible')
               line_unit_code(inline) = 0
             ENDIF           
           ENDIF
	ENDIF

* set start, delta, and N
	line_start(inline)     = lo
	line_delta(inline)     = del 
	line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1

* search for this line among the existing, dynamic lines
	outline =  TM_FIND_LIKE_DYN_LINE( inline )

	IF (outline .EQ. unspecified_int4) THEN
* ... doesn't exist yet - make a new dynamic line  "(AXnnn)"
	   CALL TM_ALLO_DYN_LINE( outline, status )
	   IF (status .NE. merr_ok) GOTO 9000
	   CALL TM_COPY_LINE( inline, outline )

* ... name it "(AXnnn)"
	   WRITE ( line_name(outline), 3000 ) outline - max_lines
 3000	   FORMAT('(AX',I3.3,')')		! restricted to 3 digits
	ENDIF

* successful completion
	status = merr_ok

* error and normal exits
 9000	CALL TM_DEALLO_DYN_LINE(inline)
	RETURN
	END
