	LOGICAL FUNCTION INHERITED_LIMITS( idim, cx )

* determine if the limits given within a definition (or within component
* variables of the definition) should be inherited by a parent definition

* for example: if A=SST[T=1-jan-1980:1-jan-1990]
* then "A+5" inherits the 1:5 info
* but A[gt=monthly_clim@mod] does NOT inherit -- the modulo regridding
* swallows up the limits information and produces, always, a full axis
* of (probably 12) values, as if no limits had been given.

* Note: this logic is also applicable to inplicit axes. For example,
* A=SST[X=160e:160w:.2] has explicit limits but the resulting new axis
* created is always provided in full, as if no limits had been given.

* However, this routine did not exist at the time that implicit axes
* were developed ... so a kludgy work-around (IS_RECONCILE(?)) was developed.


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 1/96

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, cx

* internal variable declarations:
* ...

* limits are inheritible unless modulo regridding was applied

	INHERITED_LIMITS = .NOT.cx_unstand_grid(cx)
     .			 .OR. 
     .      (cx_regrid_trans(idim,cx).LT.prgrd_modulo
     . .OR.  cx_regrid_trans(idim,cx).GT.prgrd_modmax)

	RETURN
	END


