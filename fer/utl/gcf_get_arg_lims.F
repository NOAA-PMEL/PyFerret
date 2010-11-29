	SUBROUTINE GCF_GET_ARG_LIMS (gcfcn, arg, cx_arg, res_grd,
     .				     lo_off,  hi_off, action, status)

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
* Return guidance for the passing of context information between 
* argument "arg" and the result (downstream) or default (upstream) context.
* This guidance is determined primarily from the user-specified parameters of
* the GC function (as returned by GCF_GET_ARG_PARMS), but we also watch
* here for the possibility that a dynamically imposed
* axis of the result might happen to match the axis of argument "arg".
* If so, we do pass the context information (normally no context information
* is passed for imposed axes).

* For example, in
*	LET a = RESHAPE( temp4d, out_grid )
* the result grid may share no axes in common with temp4d or (as is commonly
* the case) the result grid may differ in T and Z but match in X and Y. If
* this is the case then we do want to pass context information both upstream
* and down between the result and temp4d (just as if the axis parameter of
* the GC function  had been pgc_axis_implied_by_args and the "arg" had been
* gfcn_axis_implied_from = .TRUE. Concrete examples are
*	LIST a[x=180:160w,y=0:10n,l=1:100]
* which should pass the XY limits upstream from "a" to "temp4d" when component
* data is requested by IS_ALGEBRA. And
*	LET a2 = RESHAPE( temp4d[x=180:160w,y=0:10n], out_grid )
*	LIST a2[l=1:100]
* which should pass the XY limits of "temp4d" downstream to the result "a".

* V450: 2/97

* calling argument declarations
	INTEGER	gcfcn, arg, cx_arg, res_grd,
     .		lo_off(4),  hi_off(4), action(4), status

* internal variable declarations
	LOGICAL implied_lims(4)
	INTEGER	CGRID_AXIS, TM_LENSTR1,
     .		idim, res_ax, axis_source(4), nlen, mabstract
	CHARACTER GCF_NAME*40, name*40

#ifdef unix
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'grid_chg_fcns.parm'
	include 'xgrid_chg_fcns.cmn'
	include 'xfr_grid.cmn'       ! w/ mgrid_abstract
	include 'xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GRID_CHG_FCNS.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'       ! W/ MGRID_ABSTRACT
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif


* get the axis number for the abstract axis
	mabstract = grid_line( x_dim, mgrid_abstract )

* get the constant parameters for the GC function

c	CALL GCF_GET_AXIS_SRCS( gcfcn, 1, axis_source )
	CALL GCF_GET_AXIS_SRCS( gcfcn, arg, axis_source )

* get the constant parameters for this argument of the function
	CALL GCF_GET_ARG_PARMS
     .			(gcfcn, arg, lo_off,  hi_off, implied_lims)

* loop through the axes
	DO 500 idim = 1,4

	   res_ax  = grid_line(idim,res_grd)

	   IF (    axis_source(idim).EQ.pgc_ignore_axis) THEN
	      action(idim) = pset_to_unspec
	   ELSEIF (axis_source(idim).EQ.pgc_impose_axis) THEN

	      IF ( res_ax.EQ.mnormal
     .	      .OR. res_ax.EQ.mabstract ) THEN
	         action(idim) = pset_to_unspec

* ... We have a pgc_axis_supplied_herein axis (special axis imposed by the fcn)
	      ELSE

*     If the axis of the arg and the result don't match, don't pass context
	         IF ( res_ax .NE. CGRID_AXIS(idim,cx_arg) ) THEN
	            action(idim) = pset_to_unspec

	         ELSEIF ( implied_lims(idim) ) THEN
	            action(idim) = ppass_along

	         ELSE
	            action(idim) = pset_to_unspec

	         ENDIF
	      ENDIF

	   ELSEIF ( axis_source(idim).EQ.pgc_merge_axis) THEN
	      action(idim) = ppass_along

	   ELSE
	      name = GCF_NAME( gcfcn )
	      nlen = TM_LENSTR1( name )
	      CALL ERRMSG( ferr_internal, status,
     .			'function '//name(:nlen)
     .			//' has corrupted specs on axis '
     .			//ww_dim_name(idim), *5000 )
	   ENDIF

 500	CONTINUE

	RETURN

* error exits
 5000	RETURN
	END
