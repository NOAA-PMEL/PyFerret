	SUBROUTINE DO_POINTS(	action, idim,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

* determine the number of good and number of bad points along the given axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:   2/3/92 - based on DO_AVE_INT

* internal variable declarations:
	LOGICAL	good_un
	INTEGER i, j, k, l, pt, ngd, nbd
	REAL	bad_com, bad_res

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xunits.cmn_text'
	external xunits_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations:
	INTEGER	action, idim, com_mr, com_cx, res_mr, res_cx
	REAL    com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* --- end of introductory code ---

* initialize
        pt   = cx_lo_ss( res_cx, idim )
	bad_com = mr_bad_data( com_mr )
        bad_res = mr_bad_data( res_mr )
	good_un  = action .EQ. isact_good_pt

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO 110 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 110 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 110 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
              ngd = 0
              nbd = 0
	      DO 100 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
 100	      CONTINUE
              IF ( good_un ) THEN
                 res(pt,j,k,l) = ngd
              ELSE
                 res(pt,j,k,l) = nbd
              ENDIF
 110	   CONTINUE

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO 210 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 210 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 210 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO 200 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
 200	      CONTINUE
              IF ( good_un ) THEN
                 res(i,pt,k,l) = ngd
              ELSE
                 res(i,pt,k,l) = nbd
              ENDIF
 210	   CONTINUE

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO 310 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	   DO 310 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 310 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO 300 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
 300	      CONTINUE
              IF ( good_un ) THEN
                 res(i,j,pt,l) = ngd
              ELSE
                 res(i,j,pt,l) = nbd
              ENDIF
 310	   CONTINUE

* ALONG T AXIS
	ELSE
	   DO 410 k = cx_lo_s3(com_cx), cx_hi_s3(com_cx)
	   DO 410 j = cx_lo_s2(com_cx), cx_hi_s2(com_cx)
	   DO 410 i = cx_lo_s1(com_cx), cx_hi_s1(com_cx)
              ngd = 0
              nbd = 0
	      DO 400 l = cx_lo_s4(com_cx), cx_hi_s4(com_cx)
	         IF ( com(i,j,k,l) .EQ. bad_com ) THEN
                    nbd = nbd + 1
                 ELSE
                    ngd = ngd + 1
                 ENDIF
 400	      CONTINUE
              IF ( good_un ) THEN
                 res(i,j,k,pt) = ngd
              ELSE
                 res(i,j,k,pt) = nbd
              ENDIF
 410	   CONTINUE
	ENDIF

* success
	RETURN

* error exit
! 4000	RETURN
	END	
