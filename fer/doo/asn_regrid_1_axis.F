	SUBROUTINE ASN_REGRID_1_AXIS( cx_lims, axis, dst_lo, dst_hi,
     .				      src, msrc, dst, mdst )

* perform a regrid by "association"  (i.e. copy the data)

* 	programmer - Steve Hankin
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program

* V230:  8/22/92

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* Calling argument declarations:
	INTEGER	  cx_lims, axis, dst_lo, dst_hi, msrc, mdst
	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* Internal variable declarations:
	INTEGER i, j, k, l, idim, lo_lim(4), hi_lim(4)


* Initialize: limits for calculation
        DO 10 idim = 1, 4
           lo_lim(idim) = cx_lo_ss(cx_lims, idim)
           hi_lim(idim) = cx_hi_ss(cx_lims, idim)
 10     CONTINUE
        lo_lim(axis) = MAX( dst_lo, mr_lo_ss(msrc,axis) )
        hi_lim(axis) = MIN( dst_hi, mr_hi_ss(msrc,axis) )


* Copy the data to the destination grid
	DO 100 l = lo_lim(t_dim), hi_lim(t_dim)
	DO 100 k = lo_lim(z_dim), hi_lim(z_dim)
	DO 100 j = lo_lim(y_dim), hi_lim(y_dim)
	DO 100 i = lo_lim(x_dim), hi_lim(x_dim)
 100	dst(i,j,k,l) = src(i,j,k,l)


	RETURN
	END
