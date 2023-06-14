	SUBROUTINE HATCH_FILL_REP (zwsid)

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
*	Setup the fill area indices for hatching

*	J Davison JISAO/PMEL/TMAP
*	9.2.88/3.15.89
*	*jd* 2.14.92 mod to support atc gks
*	*jd* 10.12.92 mod for new hatching scheme


        IMPLICIT NONE
  	include 'gkspar.inc'
        include 'shade_vars.cmn'

	integer		zwsid,patt,ndx

*******************************************************************************


*	USE AREA BUNDLES IF SET
	if (area_bundles) then

*	   MISSING VALUE IS SOLID FILL
	   call FGD_GSFAR(zwsid,1,gsolid,1,1)

*	   GO THRU AVAILABLE HATCHING LEVELS 
	   do ndx = 1,shd_levels
	      patt = -1 - (ndx-1)*9/(shd_levels)
	      call FGD_GSFAR(zwsid,ndx+1,ghatch,patt,1)
	   enddo
	else
	   call FGD_GSFAIS(ghatch)
	   call FGD_GSFACI(1)
	endif

	return

	end
