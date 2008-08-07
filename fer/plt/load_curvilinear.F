	 SUBROUTINE LOAD_CURVILINEAR( xpos, ypos, dim1, dim2, argx_pos,
     .				     edges_ok, mr_xpos, mr_ypos,
     .				     cx_xpos, cx_ypos, cx_val, status )

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
* save the dimensioning information about the X position and Y position
* arrays so these arrays will be accessible from inside of PLOT+ routines.
* A C subroutine will be used to cache the pointers to these 2 arrays.
* COMMON will be used to store the size/shape information.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 3/97
* V541:  2/02 *sh - allow for subspan modulo axes
* V612: 7/08 *acm*  Fix bug 1589; see if the units of the coordinates are degrees
*                   and set a flag.

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xtext_info.cmn'
	include 'xcurvilinear.cmn'

* calling argument declarations:
	LOGICAL	edges_ok
	REAL	xpos(*), ypos(*)
	INTEGER	argx_pos, dim1, dim2,
     .		mr_xpos, mr_ypos, cx_xpos, cx_ypos, cx_val, status

* internal variable declarations:
	LOGICAL ITSA_MODULO_AXIS, ITSA_MODULO_CURVI_AXIS, TM_HAS_STRING,
     .		xspan_n_dim1, xspan_n_dim2,
     .		yspan_n_dim1, yspan_n_dim2,
     .		xspan_np1_dim1, xspan_np1_dim2,
     .		yspan_np1_dim1, yspan_np1_dim2

	INTEGER	CAXIS_MODLEN, STR_SAME,
     .		dim, x1span, x2span, y1span, y2span,
     .          span_dim1, span_dim2, i1, i2, i3
	CHARACTER*1	fieldname
	CHARACTER*64	VAR_UNITS, xcoord_units

* missing value flags
	badx = mr_bad_data(mr_xpos)
	bady = mr_bad_data(mr_ypos)

* subscript limits
	minss_dim1 = cx_lo_ss(cx_val,dim1)
	maxss_dim1 = cx_hi_ss(cx_val,dim1)
	minss_dim2 = cx_lo_ss(cx_val,dim2)
	maxss_dim2 = cx_hi_ss(cx_val,dim2)

* subscript ranges to be used for dimensioning the arguments
	xpos_loss_dim1 = mr_lo_ss(mr_xpos,dim1)
	xpos_hiss_dim1 = mr_hi_ss(mr_xpos,dim1)
	xpos_loss_dim2 = mr_lo_ss(mr_xpos,dim2)
	xpos_hiss_dim2 = mr_hi_ss(mr_xpos,dim2)
	ypos_loss_dim1 = mr_lo_ss(mr_ypos,dim1)
	ypos_hiss_dim1 = mr_hi_ss(mr_ypos,dim1)
	ypos_loss_dim2 = mr_lo_ss(mr_ypos,dim2)
	ypos_hiss_dim2 = mr_hi_ss(mr_ypos,dim2)

* offsets for start indices relative to the "values" argument indices
	xpos_off0_dim1 = cx_lo_ss(cx_xpos,dim1) - cx_lo_ss(cx_val,dim1)
	xpos_off0_dim2 = cx_lo_ss(cx_xpos,dim2) - cx_lo_ss(cx_val,dim2)
	ypos_off0_dim1 = cx_lo_ss(cx_ypos,dim1) - cx_lo_ss(cx_val,dim1)
	ypos_off0_dim2 = cx_lo_ss(cx_ypos,dim2) - cx_lo_ss(cx_val,dim2)

* determine which axes are represented as only a single point
* and which have N+1 points
	span_dim1 = cx_hi_ss(cx_val,dim1) - cx_lo_ss(cx_val,dim1) + 1
	span_dim2 = cx_hi_ss(cx_val,dim2) - cx_lo_ss(cx_val,dim2) + 1

	x1span  = cx_hi_ss(cx_xpos,dim1) - cx_lo_ss(cx_xpos,dim1) + 1
	x2span  = cx_hi_ss(cx_xpos,dim2) - cx_lo_ss(cx_xpos,dim2) + 1
	y1span  = cx_hi_ss(cx_ypos,dim1) - cx_lo_ss(cx_ypos,dim1) + 1
	y2span  = cx_hi_ss(cx_ypos,dim2) - cx_lo_ss(cx_ypos,dim2) + 1

	xspan_1_dim1   = x1span .EQ. 1
	xspan_1_dim2   = x2span .EQ. 1
	yspan_1_dim1   = y1span .EQ. 1
	yspan_1_dim2   = y2span .EQ. 1

	xspan_n_dim1 = x1span .EQ. span_dim1
	xspan_n_dim2 = x2span .EQ. span_dim2
	yspan_n_dim1 = y1span .EQ. span_dim1
	yspan_n_dim2 = y2span .EQ. span_dim2

	xspan_np1_dim1 = .NOT.(xspan_1_dim1 .OR. xspan_n_dim1)
	xspan_np1_dim2 = .NOT.(xspan_1_dim2 .OR. xspan_n_dim2)
	yspan_np1_dim1 = .NOT.(yspan_1_dim1 .OR. yspan_n_dim1)
	yspan_np1_dim2 = .NOT.(yspan_1_dim2 .OR. yspan_n_dim2)

* check for mismatched array sizes
* ... only sizes of 1, N, or N+1 are allowed
	IF (xspan_np1_dim1) THEN
	   IF (x1span .NE. span_dim1+1) THEN
	      dim = dim1
	      fieldname = 'X'
	      GOTO 5100
	   ENDIF
	ENDIF
	IF (yspan_np1_dim1) THEN
	   IF (y1span .NE. span_dim1+1) THEN
	      dim = dim1
	      fieldname = 'Y'
	      GOTO 5100
	   ENDIF
	ENDIF
	IF (xspan_np1_dim2) THEN
	   IF (x2span .NE. span_dim2+1) THEN
	      dim = dim2
	      fieldname = 'X'
	      GOTO 5100
	   ENDIF
	ENDIF
	IF (yspan_np1_dim2) THEN
	   IF (y2span .NE. span_dim2+1) THEN
	      dim = dim2
	      fieldname = 'Y'
	      GOTO 5100
	   ENDIF
	ENDIF
* ... cannot mix size of N+1 with size of N 
	IF (xspan_np1_dim1 .AND. yspan_n_dim1
     . .OR. yspan_np1_dim1 .AND. xspan_n_dim1
     . .OR. xspan_np1_dim2 .AND. yspan_n_dim2
     . .OR. yspan_np1_dim2 .AND. xspan_n_dim2) GOTO 5200

* have we been given grid box edges or grid points? 
	dim1_edges = xspan_np1_dim1 .OR. yspan_np1_dim1
	dim2_edges = xspan_np1_dim2 .OR. yspan_np1_dim2
	IF (.NOT.edges_ok) THEN
	   dim = dim1
	   IF (dim1_edges) GOTO 5300
	   dim = dim2
	   IF (dim2_edges) GOTO 5300
	ENDIF


* Have we been given modulo-able fields
* ... Both fields must agree that a dimension is modulo in order to honor it
* ... A dimension of length 1 (delta=0) is automatically modulo
	dim1_modulo = ITSA_MODULO_AXIS(cx_val,dim1)
	IF ( dim1_modulo ) THEN
	   dim1_modulo = (xspan_1_dim1
     .		     .OR. x1span .EQ. CAXIS_MODLEN(dim1,cx_xpos))
     .		.AND.
     .		         (yspan_1_dim1
     .		     .OR. y1span .EQ. CAXIS_MODLEN(dim1,cx_ypos))
     .		.AND.
     .		         ITSA_MODULO_CURVI_AXIS(xpos,ypos,mr_xpos,mr_ypos,1)
	ENDIF

	dim2_modulo = ITSA_MODULO_AXIS(cx_val,dim2)
	IF ( dim2_modulo ) THEN
	   dim2_modulo = (xspan_1_dim2
     .		     .OR. x2span .EQ. CAXIS_MODLEN(dim2,cx_xpos))
     .		.AND.
     .		         (yspan_1_dim2
     .		     .OR. y2span .EQ. CAXIS_MODLEN(dim2,cx_ypos))
     .		.AND.
     .		         ITSA_MODULO_CURVI_AXIS(xpos,ypos,mr_xpos,mr_ypos,2)
	ENDIF

* See what the units of the curvi coordinate fields are. Set a flag if the
* x coordinates are degrees.

        units_degrees = .FALSE.
        xcoord_units = VAR_UNITS (cx_xpos)
        IF  (TM_HAS_STRING(xcoord_units,'lon') .OR.
     .       TM_HAS_STRING(xcoord_units,'deg') .OR.
     .       STR_SAME(xcoord_units(1:3),'   ') .EQ. 0   ) 
     .       units_degrees = .TRUE.

* save the memory pointers to the arrays, themselves
	CALL SAVE_ARG_POINTERS( xpos, ypos )

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_inconsist_plane, status,
     .		ww_dim_name(dim)//' axis of '//fieldname
     .		//' position array', *5000 )
 5200	CALL ERRMSG( ferr_inconsist_plane, status,
     .		ww_dim_name(dim)//' axis of X and Y'
     .		//' position arrays', *5000 )
 5300	CALL ERRMSG( ferr_inconsist_plane, status,
     .		'box edges (on '//ww_dim_name(dim)//' axis)'//pCR//
     .		 'allowed only with SHADE', *5000 )

	END
