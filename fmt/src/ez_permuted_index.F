	INTEGER FUNCTION EZ_PERMUTED_INDEX( )

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
* work routine for EZ_READ when /ORDER is not the default X-Y-Z-T
* Determine the single-dimensioned array index equivalent to the 4D
* (i,j,k,l) of the "next" word to be stored

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program

* revision 0.00 -  3/4/94 *sh*

* calling arguments:
* None - all indexing is done in COMMON to maximize speed (and because
* algorithm is based on incrementing the last position rather than 
* recomputing cold from arguments

* internal variable declarations:

#ifdef unix
        include 'xez_order.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XEZ_ORDER.CMN'
#endif

* convenience equivalences
	INTEGER	nx, ny, nz,
     .		 i,  j,  k,  l,
     .		o1, o2, o3, o4
	EQUIVALENCE (eznn(1),nx), (eznn(2),ny), (eznn(3),nz),
     .		    (ezii(1),i),(ezii(2),j),(ezii(3),k),(ezii(4),l),
     .		    (ezorder(1),o1), (ezorder(2),o2), 
     .		    (ezorder(3),o3), (ezorder(4),o4) 


* local variable definitions:

* increment all 4 subscripts in appropriate permutation
	ezii(o1) = ezii(o1) + 1
	IF ( ezii(o1) .GE. eznn(o1) ) THEN
	   ezii(o1) = 0
	   ezii(o2) = ezii(o2) + 1
	   IF ( ezii(o2) .GE. eznn(o2) ) THEN
	      ezii(o2) = 0
	      ezii(o3) = ezii(o3) + 1
	      IF ( ezii(o3) .GE. eznn(o3) ) THEN
	         ezii(o3) = 0
	         ezii(o4) = ezii(o4) + 1
	      ENDIF
	   ENDIF
	ENDIF

* compute the 1D index
	EZ_PERMUTED_INDEX = ( nx*(ny*(nz*l + k) + j) + i ) + 1

	RETURN

      END
