	BLOCK DATA XCONTEXT_DATA

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
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/25/86
* revision 1.0 - 9/22/87 - added negative contexts for stored regions
* revision 1.1 - 3/10/88 - added "0:" to _ss limits
* revision 1.2 - 7/8/88  - added cx_grid to initializations
* revision 1.3 - 7/28/88 - initialize cx_stack_ptr
* revision 1.4 - 1/3/89  - added regridding transformations
* V200:  6/28/89 - eliminated _dim%,_norm,_rotation, added ss equivalences
*		 - added cx_ovsiz to get oversized regions to fill contour plots
*		 - eliminated _relative
*	12/18/89 - added cx_plot (max_context-->max_cx), cx_xtra_doc
* Ultrix/RISC port - 2/2/91 - Ultrix compiler doesnt permit variable to be
*		defined multiple times by DATA statements
*		Unix version requires implicit DO loops to be spelled out to
*		avoid duplicate initialization
*                   9/24/91 - fixed errors introduced by the above: pre-defined
*                             regions "W", "T", etc. not properly init'd
*                            (Arguably better to do this in a ferret_init file)
* V230:  1/30/92: Sun port - fixed cx_name data initialization
* V420: 9/8/95 - added initializations for implicit grid creation

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

	INTEGER	    n4cx, n4neg, num_cx_undefined, num_cx_predefined
	PARAMETER ( n4cx  = 4 * max_cx,
     .		    num_cx_predefined = 3, ! 0, -1 and -2 are named cx's T,W&N
     .		    num_cx_undefined = max_cx_predefined - num_cx_predefined,
     .		    n4neg = 4 * num_cx_undefined )
	INTEGER	    i, j

	DATA	cx_stack_ptr	/ cx_stack_ptr_base /

	DATA	cx_data_set	/ max_cx*unspecified_int4	/,
     .		cx_grid		/ max_cx*unspecified_int4	/,
     .		cx_variable	/ max_cx*unspecified_int4	/,
     .		cx_category	/ max_cx*unspecified_int4	/,
     .		cx_has_impl_grid/ max_cx*.FALSE.		/

* initialize positive subscripts
	DATA	((cx_lo_ww(i,j),i=1,4),j=1,max_cx)/ n4cx *unspecified_val8 /
	DATA	((cx_hi_ww(i,j),i=1,4),j=1,max_cx)/ n4cx *unspecified_val8 /
	DATA	((cx_lo_ss(i,j),i=1,max_cx),j=1,4)/ n4cx *unspecified_int4 /
	DATA	((cx_hi_ss(i,j),i=1,max_cx),j=1,4)/ n4cx *unspecified_int4 /
	DATA	cx_delta	/ n4cx*unspecified_val8		/
	DATA	((cx_trans(i,j),i=1,4),j=1,max_cx)
     .                                  / n4cx *trans_no_transform /
	DATA	((cx_trans_arg(i,j),i=1,4),j=1,max_cx)/ n4cx *bad_val4	 /
	DATA	((cx_by_ss(i,j),i=1,4),j=1,max_cx)/ n4cx *.FALSE.	 /
	DATA	((cx_given(i,j),i=1,4),j=1,max_cx)/ n4cx *.FALSE.	 /

* initialize undefined negative subscripts
	DATA	((cx_lo_ww(i,j),i=1,4),j=min_context,-3)
     .					/ n4neg *unspecified_val8 /
	DATA	((cx_hi_ww(i,j),i=1,4),j=min_context,-3)
     .					/ n4neg *unspecified_val8 /
	DATA	((cx_lo_ss(i,j),i=min_context,-3),j=1,4)
     .					/ n4neg *unspecified_int4 /
	DATA	((cx_hi_ss(i,j),i=min_context,-3),j=1,4)
     .					/ n4neg *unspecified_int4 /
	DATA	((cx_trans(i,j),i=1,4),j=min_context,-3)
     .					/ n4neg *trans_no_transform /
	DATA	((cx_trans_arg(i,j),i=1,4),j=min_context,-3)
     .					/ n4neg *bad_val4	 /
	DATA	((cx_by_ss(i,j),i=1,4),j=min_context,-3)
     .					/ n4neg *.FALSE.	 /
	DATA	((cx_given(i,j),i=1,4),j=min_context,-3)
     .					/ n4neg *.FALSE.	 /

	DATA
     .		cx_unstand_grid	/ max_cx*.FALSE.		/,
     .		cx_regrid_trans	/ n4cx*unspecified_int4	/

	DATA	cx_ovsiz(cx_last)/ ovsiz_off /

* pre-initialize unused named region subscripts
* (added exact limits 1/92 for Sun port)
	DATA	(cx_name(i),i=min_context,-3)/
     .		num_cx_undefined*unspecified_name4 /

* pre-initialize subscripts and ZT axes of "W", "T", and "N" regions
* (added 9/91 for Ultrix port)
	DATA	((cx_lo_ww(i,j),i=3,4),j=-2,0)
     .					/ 6*unspecified_val8 /
	DATA	((cx_hi_ww(i,j),i=3,4),j=-2,0)
     .					/ 6*unspecified_val8 /
	DATA	((cx_lo_ss(i,j),i=-2,0),j=1,4)
     .					/ 12*unspecified_int4 /
	DATA	((cx_hi_ss(i,j),i=-2,0),j=1,4)
     .					/ 12*unspecified_int4 /
	DATA	((cx_trans(i,j),i=3,4),j=-2,0)
     .					/ 6*trans_no_transform /
	DATA	((cx_trans_arg(i,j),i=3,4),j=-2,0)
     .					/ 6*bad_val4	 /
	DATA	((cx_by_ss(i,j),i=3,4),j=-2,0)
     .					/ 6*.FALSE.	 /

	DATA	cx_name(0)		/ 'W    ' /	!"W"hole GFDL region
	DATA	cx_lo_ww( x_dim, 0 )	/ 130.	/,
     .		cx_hi_ww( x_dim, 0 )	/ 290.	/,
     .		cx_lo_ww( y_dim, 0 )	/ -28.9	/,
     .		cx_hi_ww( y_dim, 0 )	/ +48.6	/,
     .		cx_by_ss( x_dim, 0 )	/ .FALSE. /,
     .		cx_by_ss( y_dim, 0 )	/ .FALSE. /,
     .		cx_trans   ( x_dim, 0 )	/ trans_no_transform /,
     .		cx_trans   ( y_dim, 0 )	/ trans_no_transform /,
     .		cx_trans_arg(x_dim, 0 )	/ bad_val4	/,
     .		cx_trans_arg(y_dim, 0 )	/ bad_val4	/,
     .		cx_given( x_dim, 0 )	/ .TRUE.  /,
     .		cx_given( y_dim, 0 )	/ .TRUE.  /,
     .		cx_given( z_dim, 0 )	/ .FALSE. /,
     .		cx_given( t_dim, 0 )	/ .FALSE. /

	DATA	cx_name(-1)		/ 'T    ' /	! "T"ropical Pacific
	DATA	cx_lo_ww( x_dim,-1 )	/ 130.	/,
     .		cx_hi_ww( x_dim,-1 )	/ 290.	/,
     .		cx_lo_ww( y_dim,-1 )	/ -23.5	/,
     .		cx_hi_ww( y_dim,-1 )	/ +23.5	/,
     .		cx_by_ss( x_dim,-1 )	/ .FALSE. /,
     .		cx_by_ss( y_dim,-1 )	/ .FALSE. /,
     .		cx_trans   ( x_dim,-1 )	/ trans_no_transform /,
     .		cx_trans   ( y_dim,-1 )	/ trans_no_transform /,
     .		cx_trans_arg(x_dim,-1 )	/ bad_val4	/,
     .		cx_trans_arg(y_dim,-1 )	/ bad_val4	/,
     .		cx_given( x_dim,-1 )	/ .TRUE.  /,
     .		cx_given( y_dim,-1 )	/ .TRUE.  /,
     .		cx_given( z_dim,-1 )	/ .FALSE. /,
     .		cx_given( t_dim,-1 )	/ .FALSE. /

	DATA	cx_name(-2)		/ 'N    '/	! "N"arrow tropical slab
	DATA	cx_lo_ww( x_dim,-2 )	/ 130.	/,
     .		cx_hi_ww( x_dim,-2 )	/ 290.	/,
     .		cx_lo_ww( y_dim,-2 )	/ -9.8	/,
     .		cx_hi_ww( y_dim,-2 )	/ +9.8	/,
     .		cx_by_ss( x_dim,-2 )	/ .FALSE. /,
     .		cx_by_ss( y_dim,-2 )	/ .FALSE. /,
     .		cx_trans   ( x_dim,-2 )	/ trans_no_transform /,
     .		cx_trans   ( y_dim,-2 )	/ trans_no_transform /,
     .		cx_trans_arg(x_dim,-2 )	/ bad_val4	/,
     .		cx_trans_arg(y_dim,-2 )	/ bad_val4	/,
     .		cx_given( x_dim,-2 )	/ .TRUE.  /,
     .		cx_given( y_dim,-2 )	/ .TRUE.  /,
     .		cx_given( z_dim,-2 )	/ .FALSE. /,
     .		cx_given( t_dim,-2 )	/ .FALSE. /


	END

