	CHARACTER*(*) FUNCTION REGRID_COMMENTS ( cx, mlen )

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

* V420:	 11/1/95 - based on VAR_TITLE_MOD code
* V450: 1/97 - corrected problem of missing blank when no units were defined

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xunits.cmn_text'
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xalgebra.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN_TEXT'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	INTEGER cx

* internal variable declarations
	LOGICAL add_on, stat_grid, non_atlin
	INTEGER	TM_LENSTR1, KNOWN_GRID,
     .		var, categ, dset, mlen, maxlen, idim, std_grid, n,
     .		axis, slen
	CHARACTER TM_FMT*16, GNAME_FROM_NUMBER*32, units*24, trans*8

* initialize
	IF ( .NOT. cx_unstand_grid(cx) ) THEN
	   REGRID_COMMENTS = ' '
	   RETURN
	ENDIF
	var     = cx_variable( cx )
	categ   = cx_category( cx )
	dset    = cx_data_set( cx )
	non_atlin = .FALSE.		! assume entirely @LIN transforms
	maxlen  = LEN( REGRID_COMMENTS )
	IF ( maxlen .LT. 12 ) GOTO 5000

* starting blank unavoidable with FORTRAN strings -- removed later
	REGRID_COMMENTS = ' '
	mlen = 1

* not appending to anything
	add_on = .FALSE.

* static (named) grids are documented by name
	stat_grid =  cx_grid(cx) .LT. max_grids
	IF ( stat_grid ) THEN
	   REGRID_COMMENTS = GNAME_FROM_NUMBER(dset, cx_grid(cx), mlen)
	   mlen = mlen + 1	! blank at end in case there's more
* ... only go on if there are transforms to report
	   DO 10 idim = 1, 4
 10	   IF (cx_regrid_trans(idim,cx).NE.prgrd_linear_interp
     .	 .AND. cx_regrid_trans(idim,cx).NE.prgrd_exact ) GOTO 50
	   RETURN
	ENDIF

* count number of axes regridded and unusual transforms
 50	std_grid = KNOWN_GRID( dset, categ, var )
* ... ... trap for internal error -- should never happen
	IF ( std_grid .EQ. unspecified_int4 ) GOTO 5000
	n = 0
	DO 100 idim = 1, 4
	   axis = grid_line(idim,cx_grid(cx))
	   IF ( axis .NE. grid_line(idim,std_grid) ) THEN
	      n = n + 1
	      non_atlin = non_atlin
     .		     .OR. cx_regrid_trans(idim,cx).NE.prgrd_linear_interp
	   ENDIF
 100	CONTINUE

* generate a string like "2 deg on X@AVE" or "Y" or "T@NGD" (for 1 axis)
*  or "Y@AVE X 5 m on Z"  or   "2 deg on X@AVE X 5 deg on Y@AVE"
	DO 120 idim = 1, 4
	   axis = grid_line(idim,cx_grid(cx))
 	   IF (axis .NE. grid_line(idim,std_grid) ) THEN
* ... lead-in string
	      IF (add_on) THEN
	         REGRID_COMMENTS = REGRID_COMMENTS(:mlen)//', '
	         mlen = MIN( maxlen, mlen+2 )
	      ELSE
	         add_on = .TRUE.
	      ENDIF
	      IF ( line_regular(axis) .AND. .NOT.stat_grid ) THEN
* ... delta
	         REGRID_COMMENTS = REGRID_COMMENTS(:mlen) //
     .				 TM_FMT(SNGL(line_delta(axis)),2,8,slen)
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
	         REGRID_COMMENTS = REGRID_COMMENTS(:mlen) //
     .				   ' '// units(:slen)//' '
	         mlen = MIN( maxlen, mlen+slen+2 )
	      ENDIF
* ... axis
	      REGRID_COMMENTS = REGRID_COMMENTS(:mlen) // 'on '
     .					//ww_dim_name(idim)
	      mlen = MIN( maxlen, mlen+4 )
* ... @transform
	      IF (non_atlin) THEN
	         trans = alg_regrid_code(cx_regrid_trans(idim,cx))
	         REGRID_COMMENTS = REGRID_COMMENTS(:mlen)//'@'//trans
	         mlen = MIN(maxlen,mlen+1+TM_LENSTR1(trans) )
	      ENDIF
	   ENDIF
 120	CONTINUE


* signal if the string has been truncated for being too long
	IF ( mlen .EQ. maxlen ) REGRID_COMMENTS(maxlen:maxlen) = '*'

* remove silly blanks from the start and end
	IF ( REGRID_COMMENTS(1:1) .EQ. ' ' ) REGRID_COMMENTS
     .					= REGRID_COMMENTS(2:maxlen)
        mlen = mlen - 1
	IF ( REGRID_COMMENTS(mlen:mlen) .EQ. ' ' ) mlen = mlen - 1

* successful completion
	RETURN

* error exit
 5000	CALL WARN( 'Regrid coruptn!!!' )
	REGRID_COMMENTS = 'regridded ???'
	RETURN

	END
