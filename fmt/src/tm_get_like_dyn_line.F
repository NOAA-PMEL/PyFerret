	SUBROUTINE TM_GET_LIKE_DYN_LINE(idim, lo, hi, del, like, class,
     .					outline, status)

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
* Returns a line matching the input description.
* Will create a line description based on the input arguments and
* compare it to other line definitions in common. Unlike
* tm_get_like_line (and unlike TM_GET_LIKE_DYN_GRID) this will check
* dynamic lines (only).
* If there is a match then "outline" returns with the number of the match.
* If there is no match, then a new dynamic line is created.
* "outline" returns with the number of the new dynamic line and its_dyn
* is .TRUE..

* created 9/95 *sh*
* Linux Port 5/96 *kob* - Had to add NINT to calculation of line_dim
*			  because the last value was sometimes getting
*			  truncated on linux, ie x[.1:1:.1] only listed
*			  nine rather than ten points.
* V533 *sh* 6/01 - save line_parent for regular stride axes, too
*		   else the stride and delta cannot be recovered 

* Arguments
	INTEGER		idim, like, class, outline, status
	REAL*8		lo, hi, del

* argument description:
* in	idim	  - axis orientation: 1-4
* in	like	  - template line for creating the new one
* in	class	  - line class
* out	outline	  - returned line
* in	lo,hi,del - description of line

* INCLUDE FILES
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'implicit.parm'
	include 'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
*
* Local definitions
	INTEGER		inline
	INTEGER		TM_FIND_LIKE_DYN_LINE

*kob 5/97
	REAL*8 		dp_epsilon
	PARAMETER 	( dp_epsilon = 1.D-10)

* allocate a scratch line 
	CALL TM_ALLO_DYN_LINE( inline, status )
	IF ( status .NE. merr_ok ) RETURN

* create a scratch line matching the request
	IF ( like .NE. unspecified_int4 ) THEN
* ... copy the template
	   CALL TM_COPY_LINE( like, inline )
	ELSE
* ... or initialize to defaults
	   line_regular(inline)    = .TRUE.
	   line_direction(inline) = axis_orients(idim)
	   line_t0(inline)        = char_init20
	   line_tunit(inline)     = 0
	   line_units(inline)     = ' '
	   line_unit_code(inline) = 0
	ENDIF
	line_name(inline) = 'scratch'
	line_class(inline) = class
	line_modulo(inline)    = .FALSE.	! modulo subset doesn't cut it

* set up the scratch line as per its class
* Note: for computational efficiency axes with regularly-spaced points will
*	be set up as "basic" axes wherever possible.  This will allow TM_WORLD
* 	to access their coordinate information without recursively descending
*	to the parent axis.  (Making potentially significant improvements
*	to commonly used routines such as ISUBSCRIPT)

	IF ( class .EQ. pline_class_basic ) THEN
	   line_parent(inline)	  = 0
	   line_regular(inline)   = .TRUE.
	   line_start(inline)     = lo
	   line_delta(inline)     = del 
	   IF ( hi .GT. 0.0D0) THEN
	     line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1
	   ELSE
   	     line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1
	   ENDIF
* ... every nth index of parent line
	ELSEIF ( class .EQ. pline_class_stride ) THEN
	   IF ( line_regular(like) ) THEN
	      line_parent(inline)    = like
	      line_start(inline)     = line_start(like)
     .				     + (lo-1)*line_delta(like)
	      line_delta(inline)     = del * line_delta(like)
	      IF ( hi .GT. 0.0D0) THEN
	        line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1
	      ELSE
   	        line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1
	      ENDIF
	   ELSE
	      line_parent(inline)    = like
	      line_start(inline)     = lo			! integer
	      line_delta(inline)     = del			! integer
	      IF ( hi .GT. 0.0D0) THEN
	        line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1
	      ELSE
   	        line_dim(inline)       = INT((hi-lo)/del +dp_epsilon) + 1
	      ENDIF
	   ENDIF

* ... midpoints from parent (entire) line
	ELSEIF ( class .EQ. pline_class_midpt ) THEN	! midpoints
	   line_dim(inline)       = line_dim(like)-1
	   IF ( line_regular(like) ) THEN
	      line_parent(inline)    = like
	      line_start(inline)     = line_start(like)
     .				     + 0.5*line_delta(like)
	      line_delta(inline)     = line_delta(like)
	   ELSE
	      line_parent(inline)    = like
	      line_start(inline)     = 1			! integer
	      line_delta(inline)     = 1			! integer
	   ENDIF

	ELSE
	   STOP 'LINE CLASS NOT IMPLEMENTED'
	ENDIF

* search for this line among the existing, dynamic lines
	outline =  TM_FIND_LIKE_DYN_LINE( inline )

	IF (outline .EQ. unspecified_int4) THEN
* ... doesn't exist yet - make a new dynamic line  "(AXnnn)"
	   CALL TM_ALLO_DYN_LINE( outline, status )
	   IF (status .NE. merr_ok) GOTO 9000
	   CALL TM_COPY_LINE( inline, outline )

* ... make sure the parent line doesn't get deallocated while still in use
	   IF ( line_parent(inline) .NE. 0 ) CALL TM_USE_LINE(like)

* ... name it "(AXnnn)"
	   WRITE ( line_name(outline), 3000 ) outline - max_lines
 3000	   FORMAT('(AX',I3.3,')')		! restricted to 3 digits

	ENDIF


! this block would apply if use count started at 1 at time of creation
!	IF (outline .NE. unspecified_int4) THEN
!* ... yes - it matches an existing dynamic line - inc its use count
!	   CALL TM_USE_LINE( outline )
!	ELSE
!
!* ... no - it shall be a new dynamic line  "(AXnnn)"
!	   CALL TM_ALLO_DYN_LINE( outline, status )
!	   IF (status .NE. merr_ok) GOTO 9000
!	   CALL TM_COPY_LINE( inline, outline )
!
!* name it "(AXnnn)"
!	   WRITE ( line_name(outline), 3000 ) outline
! 3000	   FORMAT('(AX',I3.3,')')		! restricted to 3 digits
!
!	ENDIF

* successful completion
	status = merr_ok

* error and normal exits
 9000   line_parent(inline) = 0		! avoid deallocating parent of scratch
	CALL TM_DEALLO_DYN_LINE(inline)
	RETURN
	END
