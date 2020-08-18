	SUBROUTINE ASCII_LIMITS ( cx, idim, string, slen )

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
* produce a formatted string representing the space or time limits along the
* specified axes within the specified context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/1/86
* revision 0.1 - 1/9/87  - added trans_integrated
* revision 0.2 - 4/28/87 - automated transformation comment using ALG_TRANS_TITL
* revision 0.3 - 7/13/87 - removed axis_titles - squeezed out unnedded blanks
* revision 0.4 - 9/10/87 - added "using nnn pts" for transformations w/ args
* revision 0.5 - 9/26/87 - using routine LOHI_WORLD
* revision 0.6 - 10/23/87- "using"-->"by" and nnn reduced to minimum digits
* revision 0.7 - 3/25/88 - increased LOHI_WORLD size for date/time strings
* V200:  1/12/90 - "of nnn" text string non-integer transform args
* V430:  7/96 - pass cx to ALG_TRANS_TITL so it can document 4D transformations
* v530  *acm* 3/01 - Show calendar name for alternative calendar types.
* v570  *acm* 5/04 - When showing calendar, need  cal_id .LE. max_calendars, not .LT.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction.
* v687  *acm* 3/14 Fix ticket 2149: TM_GET_CAL gets the calendar label from the T axis.
* v690 *sh* 4/14 show regridding by "false 1 pt" curvilinear regrid 
* V698 *sh* 4/16 document a true E ensemble axis with dataset identifier
* V75+ *acm*5/19 Issue 1930. Do not write member name in header unless we're 
*                looking at just one ensemble member (typo fix).
* V751 *acm*5/19 If DSG dataset, get the calendar of the time var
* V761 8/20 *acm* Longer LOHI_WORLD, LOHI_STRING strings higher-precision outputs

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xalgebra.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER		cx, idim, slen
	CHARACTER*(*)	string

* local parameter declarations:
	LOGICAL		full_precision
	PARAMETER     ( full_precision = .TRUE. )

* internal variable declarations:
	LOGICAL		TM_HAS_STRING, its_ensemble
	INTEGER		TM_LENSTR, TM_GET_CALENDAR_ID,
     .			TM_LENSTR1, ENSEMBLE_MEMBER_DSET,
     .			max_len, len_trans, llen, trans, i,
     .			grid, cal_id, len_cal, dset, slen2, memb_dset
	CHARACTER	ALG_TRANS_TITL*20, LOHI_WORLD*48, LEFINT*4, 
     .			TM_FMT*48, TM_GET_CAL*32, TM_GET_F_CAL*32,
     .			VAR_CODE*128,
     .			trans_title*128, cal_name*32
	REAL		trans_arg

* initialize
	max_len	= LEN(string)
	trans     = cx_trans( idim, cx )
	trans_arg = cx_trans_arg( idim, cx )
	dset = cx_data_set(cx)

* is this a label for an ensemble axis?
	its_ensemble = .FALSE.
	its_ensemble = idim .EQ. E_dim
     .		    .AND. dset.GT.pdset_irrelevant
     .	            .AND. dset.LE.max_gfdl_dsets
 	IF (its_ensemble) THEN
	      its_ensemble = TM_HAS_STRING(ds_type(dset), 'ENS')
	ENDIF

* not an applicable dimension ?
	IF ( cx_lo_ww( idim, cx ) .EQ. unspecified_val8 ) THEN
	   string     = ' N/A'
	   slen = 4
	   RETURN
	ENDIF

* put in lower/upper world coordinates
	string = LOHI_WORLD( idim, cx, ' to ', full_precision, slen )
	
* transformation requiring comment ?
	IF ( trans .NE. trans_no_transform ) THEN
	   trans_title = ALG_TRANS_TITL( trans, cx )
	   len_trans   = TM_LENSTR( trans_title )
	   IF ( trans_arg .NE. bad_val4 ) THEN
	      DO 100 i = 1, num_transforms
 100	      IF ( trans .EQ. alg_trans_num(i) ) GOTO 200
	      STOP 'ASCII_LIMITS'
 200	      IF ( alg_trans_its_int(i) ) THEN
* ... add "by nnn pts" squeezing "nnn" to minimum digits
	         trans_title = trans_title(:len_trans)//' by '//
     .			       LEFINT( INT(trans_arg), llen )
	         len_trans = len_trans + 4 + llen
	         trans_title = trans_title(1:len_trans)//' pts'
	         len_trans = len_trans + 4
	      ELSE
* ... add "of nnn " in a nice format
	         trans_title = trans_title(:len_trans)//' of '//
     .			       TM_FMT( trans_arg, 4, 10, llen )
	         len_trans = len_trans + 4 + llen
	      ENDIF
	   ENDIF
	   string = string(1:slen)//' ('//trans_title(:len_trans)//')'	
	   slen = MIN( max_len, slen+len_trans+3 )
	ELSEIF (cx_aux_stat(idim,cx) .EQ. paux_stat_false_1pt_regrid) THEN
* ... add "averaged using DEPTH" or "interpolated using DEPTH"
	   IF     (cx_regrid_trans(idim, cx) .EQ. pauxrgrd_linear ) THEN
	      trans_title = 'interpolated using'
	   ELSEIF (cx_regrid_trans(idim, cx) .EQ. pauxrgrd_pwlave ) THEN
	      trans_title = 'pc wise lin avgd using'
	   ELSE
	      trans_title = '??crptn??'
	   ENDIF	 
	   len_trans = TM_LENSTR( trans_title )
*      append name of auxiliary variable
	   trans_title(len_trans+2:) = VAR_CODE(cx_aux_cat(idim,cx),
     .				                cx_aux_var(idim,cx)  )
	   len_trans = TM_LENSTR( trans_title )
	   string = string(1:slen)//' ('//trans_title(:len_trans)//')'	
	   slen = MIN( max_len, slen+len_trans+3 )
	ENDIF

* calendar name requiring comment ?  If DSG dataset, get the calendar of the time var


	IF ((idim.EQ.t_dim .OR. idim.EQ.f_dim) .AND. 
     .      cx_lo_ww( idim, cx ) .NE. unspecified_val8 ) THEN

	   grid = cx_grid(cx)

	   IF (dset .GT. pdset_irrelevant) THEN
	      IF (dsg_ragged(dset)) grid = dsg_xlate_grid(dset) 
	   ENDIF

	   IF (idim.EQ.t_dim) cal_name = TM_GET_CAL (grid)
	   IF (idim.EQ.f_dim) cal_name = TM_GET_F_CAL (grid)
	   cal_id = TM_GET_CALENDAR_ID (cal_name)
	   IF (cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
	      len_cal = TM_LENSTR(cal_name)
	      string =  string(1:slen)//' '//cal_name(:len_cal)
	      slen = MIN( max_len, slen+len_cal+1 )
	   ENDIF
	ENDIF

* a single ensemble member name
	IF ( its_ensemble
     .  .AND.cx_lo_ss(cx,idim).NE.unspecified_int4
     .  .AND.cx_lo_ss(cx,idim).EQ.cx_hi_ss(cx,idim) ) THEN

* ... get the special string that indicates the ensemble member
	   memb_dset = ENSEMBLE_MEMBER_DSET(dset, cx_lo_ss(cx,idim))
	   IF (  memb_dset.GT.pdset_irrelevant
     .	    .AND.memb_dset.LE.max_gfdl_dsets) THEN
	      slen2 = TM_LENSTR1( ds_name(memb_dset) )
	      slen2 = MIN(slen2, 12)  ! no more than 12 chars
	      string = string(1:slen)//' ('// 
     .		       ds_name(memb_dset)(1:slen2) // ')'	
	      slen = TM_LENSTR1( string )
	   ENDIF
	ENDIF


	RETURN
	END
