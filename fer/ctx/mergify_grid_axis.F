	SUBROUTINE MERGIFY_GRID_AXIS( cx, idim, uvar )

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
* Determine the nature of how a particular axis of a requested user variable
* depends on the context region and initialize uvar_given for the user var.
* The component passed (cx) is assumed to be the first that contains relevant
* information about this axis.
* For example "let a=i[i=1:5]" has i fully specified and needs no outside
* information, whereas "list i" or "list u[Z=@ave]" can be meaningful only if
* external region information is supplied 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  4/16/90 - extracted from MERGIFY_GRID
* V230:  1/27/92 - changed uvlim_*_hi (unused) to uvlim_*_lohi to describe
*                  documentation changed, only
* V420:  11/95   - was incorrectly giving uvlim_need_xact for b = a[i=@max]
*			where a = temp[i=1:160]
	
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations
	INTEGER	cx, idim, uvar

* local variable declarations
	LOGICAL INHERITED_LIMITS, itsa_uvar, itsa_pvar,
     .		lims_with_component
	INTEGER	REGION_CLASS, NEST_REGION_CLASS, class, cat, var, pvdim

* There are subtlities associated with limits that are given in the UVAR
* definition.  These are classified in uvar_given under these codings:
*	uvlim_irrelevant- limits are irrelevant: NEW=10.1
*	uvlim_gvn_gnl	- limits given in a gen'l way: NEW=U[K=1:5]
*	uvlim_gvn_xact	- limits that reduce to a pt.: NEW=U[K=1:5@AVE]
*	uvlim_needed	- no limit info. given
*	uvlim_need_lo	- no lims but transf. needs xact lo:NEW=U[K@IIN]
*	uvlim_need_lohi - no lims but transf. needs both xact:NEW=U[K@WEQ]
*	uvlim_need_xact - no lims but transf. needs both xact:NEW=U[K@AVE]
* if a component doesn't have explicit limits (like "k" in X=U[K=1]*2)
* then the user must specify some amount of limit information to access
* the defined variable

* complications arise through the nesting of limit information.  For example,
* DEFINE VARIABLE B = A * 10  where  DEFINE VARIABLE A=TEMP[K=@AVE]
* When we "MERGIFY" the grid for B the first component is another user-variable

* initialize
	cat = cx_category(cx)
	var = cx_variable(cx)
	itsa_uvar = cat .EQ. cat_user_var
	itsa_pvar = cat .EQ. cat_pseudo_var
	IF (itsa_pvar ) pvdim = MOD( var-1, 4 ) + 1
	lims_with_component = .FALSE.

* is the component a user-defined variable ?
	IF ( itsa_uvar ) THEN

* ... first component is a user-variable with this axis irrelevant
	   IF ( uvar_given(idim,var) .EQ. uvlim_irrelevant ) GOTO 100

	   class = NEST_REGION_CLASS( idim, cx, cx_variable(cx) )

! lims_with_component added 11/95
! 11/95 - needed for dynamic axes (?? - comment 1/96)
!  1/96 - modified via  
! Note that this logic may be wrong in subtle but important ways:
!	the logic of uvar_given was originally conceived to determine
!	if a result cached in memory could be reused. It is now being
!	to alter the interpretation of limits DURING calculation
!	... should some of this logic be applied to cx_given instead?
! Note 2: non-INHERITIBLE limits **always** go with class = gnl
!		(not xact, lo, hi)
	   lims_with_component = uvar_given(idim,var).LT.uvlim_needed
     .			    .AND.INHERITED_LIMITS( idim, cx )

* pseudo-variables are only relevant along their defining axes
	ELSEIF ( itsa_pvar .AND. (idim.NE.pvdim) ) THEN
	   GOTO 100

	ELSE
	   class = REGION_CLASS( idim, cx )
	ENDIF

* limits are needed if they were not supplied either with the
* transformation (e.g. U[I=1:5@ave]) or with the underlying component
* (e.g. U2[I=@ave] where U2=U[I=1:5])
!pre 11/95 IF ( cx_given(idim,cx) ) THEN
	IF ( cx_given(idim,cx) .OR. lims_with_component ) THEN
	   uvar_given(idim,uvar) = class
	ELSE
	   uvar_given(idim,uvar) = class + uvlim_needed
	ENDIF

* successful completion
 100	RETURN
	END
