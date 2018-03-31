	SUBROUTINE SET_LEVELS_CENTERKEY (do_shift)


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
* ACM 3/22/2018
* issue 1607: For centered labels draw the plot so that the levels are
* describing the middle of the colors not the contour lines between colors.
* Interpolate /LEVELS values to obtain new '''edge-values''' for the contours 
* and patch-edges; then SHADE and CONTOUR just as we always have, but using 
* those new computed levels. The color key would labeled at the patch-centers 
* using the original /LEVELS values. 
*
* When plotting is done,  will restore the previous levels, so any plot using /LEVELS
* without an argument will start this process again.
* If do_shift is FALSE, undo this shifting, restoring previous levels an nlev2.

	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'cont_inc.decl'
	include		'CONT.INC'
        include         'shade_vars.cmn'

	LOGICAL do_shift

	REAL zh, delz, vlo, vhi

	IF (.NOT.center_key_labels) RETURN

	IF (do_shift) THEN

	   nlev2_save = nlev2
	   DO i = 1, nlev2
	      zlevsave(i) = zlev(i) 
	   ENDDO

* When the palette is by_value, the color-spectrum setting needs to
* correspond to the original levels, but with an extra upper-bound
* If there is pos-inf, it lies below that value.

	   IF (rgb_mapping .eq. by_value) THEN
	      IF (posinf) THEN
	         del = zlev(nlev-1) - zlev(nlev-2)
	         nlev = nlev + 1
	         zlev(nlev-1) = zlev(nlev-1) + del
	      ELSE
	         del = zlev(nlev) - zlev(nlev-1)
	         nlev = nlev + 1
	         zlev(nlev) = zlev(nlev-1) + del
	      ENDIF
	      nlev2 = nlev
	      RETURN

	   ENDIF

	   zh = zlev(nlev2)
           IF (neginf) zh = zlev(nlev2-1)
	   i1 = 1
	   in = nlev2
	   IF (neginf) i1 = 2
	   IF (posinf) in = nlev2-1

*    ... initial lower bound 

	   vlo = zlevsave(i1)
	   vhi = zlevsave(i1+1)
	   zlev(i1) = vlo - (vhi-vlo)/2.

*    ... interpolate between middle levels

	   DO i = i1+1, in
	      vlo = zlevsave(i-1)
	      vhi = zlevsave(i)
	      zlev(i) = vlo + (vhi-vlo)/2.
	   ENDDO

*    ... upper-most bound
           zlev(in+1) = vhi + (vhi-vlo)/2.

           IF (neginf) zlev(1) = zlev(2) - 1
           IF (posinf) zlev(in+2) = zlev(in+1) + 1

	   nlev2 = nlev2+1

* Or, restore un-shifted set of levels

	ELSE
	   nlev2 = nlev2_save
	   nlev  = nlev2_save
	   DO i = 1, nlev2
	      zlev(i) = zlevsave(i) 
	   ENDDO
	ENDIF

	RETURN
	END
