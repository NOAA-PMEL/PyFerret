	SUBROUTINE PLOT_ALONG_DATA_SET_UP( status )

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
* modify the interpretation of the command data so it looks like
* a collection of 1D lines in the "/ALONG" direction
* this will be accomplished by generating a list of contexts, each for a
* single line, each pointing to the same, shared memory-resident array

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V698 *sh* 3/16

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'command.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xplot_setup.cmn'
	include 'xtext_info.cmn'
	
* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	LOGICAL has_keys
	INTEGER	CX_DIM_LEN, iline, nlines, stride(nferdims),
     .		ss(nferdims), ssmin(nferdims), ssmax(nferdims),
     .		idim, along_dim, the_dim, ndim, dim(nferdims),
     .		max_along_lines, cx0, the_cx, nreduce_axes, loc, slen
	REAL	reduce_factor
	CHARACTER*1 buff1, LEFINT*8, buff8*8

* initialize
	DO idim = 1, nferdims
	   stride(idim) = 1
	ENDDO

* get the desired orientation   e.g. "/ALONG=T"
	loc = qual_given( slash_plot_along )
	CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                     buff1, status )
	IF (status .NE. ferr_ok) GOTO 5000
	DO along_dim = 1, nferdims
	   IF ( buff1 .EQ. ww_dim_name(along_dim) ) GOTO 100
	ENDDO
* ... unrecognized orientation	         
	GOTO 5100

* can only be a single argument in this form of the command
 100	IF (num_uvars_in_cmnd .GT.1) GOTO 5200

* how many individual lines have been requested?
	CALL TRANSFER_CONTEXT( is_cx(1), cx_buff )
	CALL GET_CX_DIMS(cx_buff, ndim, dim)
	nlines = 1
	DO idim = 1, ndim
	   the_dim = dim(idim)
	   IF (the_dim .EQ. along_dim) CYCLE
	   nlines = nlines * CX_DIM_LEN(the_dim, cx_buff)
	ENDDO
	IF (nlines .EQ. 1) RETURN   ! no action needed here

* have plot keys been requested?
	has_keys = qual_given(slash_plot_nokey) .EQ. 0

* how many individual lines are allowed
* (max_key_entries smaller than max_line_on_plot)
	max_along_lines = max_noframe_context - is_cx(1) + 1
	IF (max_along_lines .GT. max_line_on_plot) max_along_lines =
     .						   max_line_on_plot
	IF (has_keys) THEN
	   max_along_lines = max_key_entries
	   IF (nlines .GT. max_key_entries) THEN
	      buff8 = LEFINT( max_key_entries,  slen  )
	      CALL WARN('Decimating PLOT/ALONG lines to /KEY limit of '
     .			// buff8(:slen) )
	   ENDIF
	ELSEIF (nlines .GT. max_along_lines) THEN
	   buff8 = LEFINT( max_along_lines,  slen  )
	   CALL WARN('Decimating PLOT/ALONG lines to /NOKEY limit of '
     .			// buff8(:slen) )
	ENDIF

* if this exceeds the max allowed, figure out how to stride the other axes
	IF (nlines .GT. max_along_lines) THEN
	   reduce_factor = FLOAT(nlines) / max_along_lines
	   nreduce_axes = ndim - 1   ! no striding on the /ALONG axis
* ... an axis with only 2 points cannot be strided
	   DO idim = 1, ndim
	      the_dim = dim(idim)
	      IF (the_dim .EQ. along_dim) CYCLE
	      IF (CX_DIM_LEN(the_dim,cx_buff).EQ.2) nreduce_axes =
     .		nreduce_axes - 1	 
	   ENDDO
* ... all axes of len>2 get the same stride
	   reduce_factor = reduce_factor ** (1./nreduce_axes)
	   reduce_factor = INT(reduce_factor+0.9999)  ! round up
	   DO idim = 1, ndim
	      the_dim = dim(idim)
	      IF (the_dim .EQ. along_dim) CYCLE
	      IF (CX_DIM_LEN(the_dim,cx_buff).EQ.2) CYCLE
	      stride(the_dim) = reduce_factor
	   ENDDO
* ... what is the resulting number of lines?
	   nlines = 1
	   DO idim = 1, ndim
	      the_dim = dim(idim)
	      IF (the_dim .EQ. along_dim) CYCLE
	      nlines = nlines *
     .		       INT(CX_DIM_LEN(the_dim,cx_buff)/stride(the_dim))
	   ENDDO
	   IF (nlines .GT. max_along_lines) GOTO 5300
	ENDIF

* initiailize the axis subscript counters
	DO idim = 1, ndim
	   the_dim = dim(idim)
	   IF (the_dim .EQ. along_dim) CYCLE
	   ssmin(the_dim) = cx_lo_ss(cx_buff,the_dim)
	   ssmax(the_dim) = cx_hi_ss(cx_buff,the_dim)
	   ss(the_dim)    = ssmin(the_dim)
	ENDDO

* populate the contexts
	cx0 = is_cx(1) - 1
	DO iline = 1, nlines

	   is_mr(iline) = is_mr(1)  ! a single mr shared by all
! removed cuz num_uvar_in_cmnd is reset to 1 at end of XEQ_PLOT. Have to reset
! it because when same identical expression is found in NEXT command,
! GET_PROT_CMND_DATA declines to re-parse it, so that EX#1 can remain in memory
!	   CALL MR_IN_USE(is_mr(1))

	   the_cx = cx0 + iline
	   is_cx(iline) = the_cx
	   CALL TRANSFER_CONTEXT( cx_buff, the_cx )

* ... set the per-axis positions up
	   DO idim = 1, ndim
	      the_dim = dim(idim)
	      IF (the_dim .EQ. along_dim) CYCLE
	      cx_by_ss(the_dim, the_cx) = .TRUE.
	      cx_lo_ss(the_cx,the_dim) = ss(the_dim)
	      cx_hi_ss(the_cx,the_dim) = ss(the_dim)
	      CALL FLESH_OUT_AXIS( the_dim, the_cx, status )
	      IF (status .NE. ferr_ok) GOTO 5000
	   ENDDO

* ... prepare for the next context
	   DO idim = 1, ndim
	      the_dim = dim(idim)
	      IF (the_dim .EQ. along_dim) CYCLE
	      ss(the_dim) = ss(the_dim) + stride(the_dim)
	      IF (ss(the_dim) .LE. ssmax(the_dim)) THEN
	         EXIT
	      ELSE
	         ss(the_dim) = ssmin(the_dim)
	         CONTINUE     ! continue to the next dimension
	      ENDIF
	   ENDDO
	ENDDO

* un-bump by one, for the original cx, which has been over-written
! commented out along with CALL MR_IN_USE above
!	CALL MR_NOT_IN_USE(is_mr(1))

* make the rest of Ferret believe that all these lines were asked for
	num_uvars_in_cmnd = nlines

* successful completion
	RETURN

* error exit2
 5000	RETURN
 5100	CALL ERRMSG(ferr_invalid_command, status,
     .          "PLOT/ALONG='axis', where axis = X,Y,Z,T,E or F", *5000)
 5200	CALL ERRMSG(ferr_invalid_command, status,
     .          "Use just a single argument with PLOT/ALONG", *5000)
 5300	CALL ERRMSG(ferr_internal, status,
     .		"plot_along_data_set_up strides ??!!", *5000)

	END




