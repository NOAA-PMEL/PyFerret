	CHARACTER*(*) FUNCTION REGRID_COMMENTS ( cx, mlen )

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
* return text describing the regridding done

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:	 11/1/95 - based on VAR_TITLE_MOD code
* V450: 1/97 - corrected problem of missing blank when no units were defined
* V510: 4/00 *sh* - netCDF grids are now dynamic, so "(" as first character
*		    of name is basis for not using the associated variable
* V533: 6/01 *sh* - @XACT doesn't need a comment if for xact strides
* V552  4/03 *acm*  up gname_from_number to 128 characters (variable names)
* V671  6/11 *acm*  New 2D linear xy regridding, fixing ticket 1862
* V68  *acm* 1/12  changes for double-precision ferret.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V690 *sh* 1/14 document sigma and curvilinear aux var regridding
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* V698  1/16 *sh*  special formatting of T-F plane FMRC regridding comments
* V741  5/18 *acm* issue 1868: gridding transforms that will grid to same axis

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'implicit.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include	'xtext_info.cmn'

* calling argument declarations
	INTEGER cx

* internal variable declarations
	LOGICAL REGRID_TO_SELF,
     .		add_on, stat_grid, non_atlin, lin_regrid, 
     .		did_regrid(nferdims)
	INTEGER	TM_LENSTR1, KNOWN_GRID,
     .		var, categ, dset, mlen, maxlen, idim, std_grid, n,
     .		axis, slen
	CHARACTER TM_FMT*8, GNAME_FROM_NUMBER*128, VAR_CODE*128,
     .		  ALG_TRANS_TITL*20, units*24, trans*8, mybuff*128,
     .		  sep*1, buff*128

* initialize
	IF ( .NOT. cx_unstand_grid(cx) ) THEN
	   buff = ' '
	   mybuff = ' '
	   RETURN
	ENDIF
	var     = cx_variable( cx )
	categ   = cx_category( cx )
	dset    = cx_data_set( cx )
	non_atlin = .FALSE.		! assume entirely @LIN transforms
	maxlen  = LEN( buff )
	IF ( maxlen .LT. 12 ) GOTO 5000
	stat_grid =  grid_name(cx_grid(cx))(1:1) .NE. '('

* starting blank unavoidable with FORTRAN strings -- removed later
	buff = ' '
	mybuff = ' '
	mlen = 1

* not appending to anything
	add_on = .FALSE.

* static (named) grids are documented by name
	IF ( stat_grid ) THEN
	   buff = GNAME_FROM_NUMBER(dset, cx_grid(cx), mlen)
	   mybuff = GNAME_FROM_NUMBER(dset, cx_grid(cx), mlen)
	   mlen = mlen + 1	! blank at end in case there's more
* ... only go on if there are transforms to report
	   DO 10 idim = 1, nferdims
	     IF (cx_regrid_trans(idim,cx) .NE. prgrd_linear_interp
     .	   .AND. cx_regrid_trans(idim,cx) .NE. prgrd_exact
     .	   .AND. cx_regrid_trans(idim,cx) .NE. prgrd_xy_lin) THEN
	       GOTO 50
	     ENDIF
 10	   CONTINUE
	   GOTO 900
	ENDIF

* count number of axes regridded and unusual transforms
 50	std_grid = KNOWN_GRID( dset, categ, var )
* ... ... trap for internal error -- should never happen
	IF ( std_grid .EQ. unspecified_int4 ) GOTO 5000
	n = 0
	DO 100 idim = 1, nferdims
	   axis = grid_line(idim,cx_grid(cx))
	   did_regrid(idim) = ( axis .NE. grid_line(idim,std_grid) ) .OR. 
     .        REGRID_TO_SELF (cx_regrid_trans(idim, cx) )
	   IF ( did_regrid(idim) ) THEN
	      n = n + 1
	      lin_regrid = ( (cx_regrid_trans(idim,cx).EQ.prgrd_linear_interp) .OR. 
     .             (cx_regrid_trans(idim,cx).EQ.prgrd_xy_lin) )
	      non_atlin = non_atlin
     .	      .OR. ( .NOT. lin_regrid 
     .		.AND. .NOT.( cx_regrid_trans(idim,cx) .EQ. prgrd_xact_pts
     .	               .AND. line_class(axis) .EQ. pline_class_stride ) )
	   ENDIF
 100	CONTINUE

* generate a string like "2 deg on X@AVE" or "Y" or "T@NGD" (for 1 axis)
*  or "Y@AVE X 5 m on Z"  or   "2 deg on X@AVE X 5 deg on Y@AVE"
	DO 120 idim = 1, nferdims
	   axis = grid_line(idim,cx_grid(cx))
 	   IF (axis .NE. grid_line(idim,std_grid) .OR. did_regrid(idim)) THEN
* ... lead-in string
	      IF (add_on) THEN
	         buff = buff(:mlen)//', '
	         mybuff = mybuff(:mlen)//', '
	         mlen = MIN( maxlen, mlen+2 )
	      ELSE
	         add_on = .TRUE.
	      ENDIF

	      IF (cx_regrid_trans(idim,cx) .EQ. pauxrgrd_fmrc ) THEN
* ... comment about T-F axis reorganization for FMRC data sets:
*          "time series view using 2d TIME2D field"
	         IF (idim .EQ. t_dim) THEN
     	            buff = buff(:mlen) //
     .                  'time series view'
	            mlen = mlen + 16
	         ELSEIF (idim.EQ. f_dim) THEN
     	            buff = buff(:mlen) //
     .                  'forecast visualization view'
	            mlen = mlen + 27
	         ENDIF
	      ELSE

	         IF ( line_regular(axis) .AND. .NOT.stat_grid 
     .	    .   AND.  line_dim    (axis) .GT.  1               ) THEN
* ... delta
#ifdef double_p
	            buff = buff(:mlen) //
     .				 TM_FMT(line_delta(axis),2,8,slen)
	            mybuff = mybuff(:mlen) //
     .				 TM_FMT(line_delta(axis),2,8,slen)
#else
     	            buff = buff(:mlen) //
     .				 TM_FMT(SNGL(line_delta(axis)),2,8,slen)
	            mybuff = mybuff(:mlen) //
     .				 TM_FMT(SNGL(line_delta(axis)),2,8,slen)
#endif
	            mlen = MIN( maxlen, mlen+slen )
* ... units
	            IF ( line_units(axis) .EQ. ' ' ) THEN
	               units = 'delta'
	            ELSE
	               IF ( line_unit_code(axis) .EQ. 0 ) THEN
	                  units = line_units(axis)
	               ELSE
	                  units = un_name(line_unit_code(axis))
	               ENDIF
	            ENDIF
	            slen = TM_LENSTR1(units)
	            buff = buff(:mlen) //
     .				   ' '// units(:slen)//' '
	            mybuff = mybuff(:mlen) //
     .				   ' '// units(:slen)//' '
	            mlen = MIN( maxlen, mlen+slen+2 )
	         ENDIF
* ... axis
	         buff = buff(:mlen) // 'on '
     .					//ww_dim_name(idim)
	         mybuff = mybuff(:mlen) // 'on '
     .					//ww_dim_name(idim)
	         mlen = MIN( maxlen, mlen+4 )
* ... @transform
	         IF (non_atlin) THEN
	            trans = alg_regrid_code(cx_regrid_trans(idim,cx))
	            IF ( trans(:4) .NE. unspecified_name4 ) THEN
	              buff = buff(:mlen)//'@'//trans
	              mybuff = mybuff(:mlen)//'@'//trans
	              mlen = MIN(maxlen,mlen+1+TM_LENSTR1(trans) )
	            ENDIF
	         ENDIF
	      ENDIF
	   ENDIF
 120	CONTINUE

* layerz and curvilinear regridding
* if an auxiliary variable regrid then document "using DEPTH" or whatever
 900	IF (cx_naux(cx) .GT. 0) THEN
	   mybuff = ' using'
	   slen = 6
	   sep = ' '
	   DO idim = 1, nferdims
	      IF (cx_regrid_trans(idim,cx) .EQ. pauxrgrd_fmrc ) THEN
	         IF (idim .EQ. f_dim ) CYCLE
	         mybuff = mybuff(:slen) // ' 2d'
	         slen = slen + 3
	      ENDIF
	      IF (cx_aux_cat(idim,cx) .NE. unspecified_int4) THEN
	         mybuff = mybuff(:slen) // sep //
     .			  VAR_CODE(cx_aux_cat(idim,cx),
     .				   cx_aux_var(idim,cx)  )
	         slen = TM_LENSTR1(mybuff)
	         sep = ','
	      ENDIF
	   ENDDO
	   buff = buff(:mlen)//mybuff
	   mlen = MIN(maxlen,mlen+slen)
	   buff = buff(:mlen)//' field'
	   mlen = MIN(maxlen,mlen+6)
	   IF (cx_naux(cx) .GT. 1) THEN
	      buff = buff(:mlen)//'s'
	      mlen = MIN(maxlen,mlen+1)
	   ENDIF
	ENDIF

* signal if the string has been truncated for being too long
	IF ( mlen .EQ. maxlen ) buff(maxlen:maxlen) = '*'
	IF ( mlen .EQ. maxlen ) mybuff(maxlen:maxlen) = '*'

* remove silly blanks from the start and end
	IF ( buff(1:1) .EQ. ' ' )
     .     buff = buff(2:maxlen)
	IF ( mybuff(1:1) .EQ. ' ' )
     .     mybuff = mybuff(2:maxlen)
        mlen = mlen - 1
	IF ( buff(mlen:mlen) .EQ. ' ' ) mlen = mlen - 1
	IF ( mybuff(mlen:mlen) .EQ. ' ' ) mlen = mlen - 1

* successful completion
	REGRID_COMMENTS = buff
	RETURN

* error exit
 5000	CALL WARN( 'Regrid coruptn!!!' )
	REGRID_COMMENTS = 'regridded ???'
	RETURN

	END
