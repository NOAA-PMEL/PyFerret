	CHARACTER*(*) FUNCTION KEY_STRING( cx, flags, slen )

* generate a string that describes the given context using the flags as
* guidelines for what limits, transformations, etc. to document

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200: 1/2/89
* V230:  11/4/92 - expanded variable name length
* V420:	 9/25/95 - comments appropriate to dynamic grids

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

* calling argument declarations
	LOGICAL	flags(0:10)
	INTEGER	cx, slen

* local variable declarations:
	LOGICAL		save_by_ss
	INTEGER		TM_LENSTR1, idim, maxlen, plen, categ, var, dset
	CHARACTER 	ALG_TRANS_CODE*3, ARG_STRING*12, VAR_CODE*24,
     .			VAR_TITLE*64, CX_DIM_STR*40, REGRID_COMMENTS*64,
     .			prefix*1, dim_name*1, name*48

* local parameter declarations:
	LOGICAL		full_prec
	PARAMETER     ( full_prec = .TRUE. )

* initialize
	var     = cx_variable( cx )
	categ   = cx_category( cx )
	dset    = cx_data_set( cx )
	prefix = '['
	maxlen = LEN( KEY_STRING )

* begin with the code for this variable	
	KEY_STRING = VAR_CODE( categ, var )
	IF ( INDEX(KEY_STRING,'#') .GT. 0 ) KEY_STRING = VAR_TITLE ( cx )
	slen = TM_LENSTR1( KEY_STRING )

* document the data set
	IF ( flags(pdoc_dset) .AND. cx_data_set(cx).NE.pdset_irrelevant ) THEN
	   name = ds_name( cx_data_set(cx) )
	   plen = TM_LENSTR1(name)
	   KEY_STRING = KEY_STRING(:slen)//prefix//'D='//name(:plen)
	   slen = MIN( maxlen, slen+plen+3 )
	   prefix = ','
	ENDIF	   

* document any regridding
* (as of FERRET v2.00 the flag for this is ignored)
	IF ( cx_unstand_grid(cx) ) THEN
	   KEY_STRING = KEY_STRING(:slen)//prefix//'G='//
     .			REGRID_COMMENTS(cx, plen)
	   slen = MIN( maxlen, slen+3+plen )
	   prefix = ','
	ENDIF

* document each axis (and it's transformation) that's requested
	DO 100 idim = 1, 4
* ... all keys in world coordinates
	   save_by_ss = cx_by_ss( idim, cx )
	   cx_by_ss( idim, cx ) = .FALSE.
	   IF ( flags(idim)
     .    .AND. cx_lo_ww(idim,cx) .NE. unspecified_val8
     .    .AND. cx_lo_ww(idim,cx) .NE. unspec_val8_2	) THEN
	      KEY_STRING = KEY_STRING(:slen)//prefix//
     .				CX_DIM_STR(idim,cx,':',full_prec,plen)
	      slen = MIN( maxlen, slen+plen+1 )
	      prefix = ','
	   ELSEIF ( flags(idim+pdoc_offset_xform)
     .    .AND. cx_trans(idim,cx) .NE. trans_no_transform
     .    .AND. cx_trans(idim,cx) .NE. unspecified_int4 ) THEN
	      IF ( cx_by_ss( idim, cx ) ) THEN
	         dim_name = ss_dim_name( idim )
	      ELSE
	         dim_name = ww_dim_name( idim )
	      ENDIF
	      KEY_STRING = KEY_STRING(:slen)//prefix//dim_name//'=@'//
     .				ALG_TRANS_CODE( cx_trans( idim, cx )) //
     .				ARG_STRING( cx_trans_arg( idim, cx ))
	      slen = TM_LENSTR1( KEY_STRING )
	      prefix = ','
	   ENDIF
* ... restore " by subscript" info
	   cx_by_ss( idim, cx ) = save_by_ss
 100	CONTINUE

* clean up
	IF ( prefix .EQ. ',' ) THEN
	   KEY_STRING = KEY_STRING(:slen)//']'
	   slen = MIN( maxlen, slen+1 )
	ENDIF
	IF ( slen .EQ. maxlen ) KEY_STRING(slen:slen) = '*'

	RETURN
	END
