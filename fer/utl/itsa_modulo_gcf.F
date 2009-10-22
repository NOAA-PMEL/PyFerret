	LOGICAL FUNCTION ITSA_MODULO_GCF( cx, idim )

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
* determine whether a particular axis of a grid changing variable should have
* modulo applied to the RESULT of the function rather than to the arguments,
* as is normal for non-GC uvars

* This function ASSUMES that the passed context refers to a grid-changing uvar

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 2/97
*       97.07.16 *jc* adding code to query external functions
* V63  *acm* 10/09 Changes for gfortran build

	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xgrid_chg_fcns.cmn'

* calling argument declarations:
	INTEGER	cx, idim, axis_will_be(4)

* internal variable declarations:
	INTEGER	 WHICH_GCFCN, gcfcn

* get grid-changing function
	gcfcn = WHICH_GCFCN( cx_variable(cx) )

* get axis sources
	IF ( gcfcn .GT. gfcn_num_internal ) THEN

	   call efcn_get_axis_will_be(gcfcn, axis_will_be)
	   ITSA_MODULO_GCF = axis_will_be(idim)
     .	           .EQ. pgc_axis_supplied_herein 

	ELSE

	   ITSA_MODULO_GCF = gfcn_axis_will_be(idim,gcfcn)
     .	           .EQ. pgc_axis_supplied_herein 

	ENDIF

	RETURN
	END
