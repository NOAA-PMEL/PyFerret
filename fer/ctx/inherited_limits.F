	LOGICAL FUNCTION INHERITED_LIMITS( idim, cx )

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


