	INTEGER FUNCTION GCF_PARENT_CHAR( child_uvar, start )

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
* Given the starting character number of this uvar within it immediate parent
* return the starting character number of this uvar within its ancestor

*    Note that the ancestor may be a grand-parent or great-grand
*    For example, in the expression
*        A = FCN1(v1+v2) + B
*    where FCN1 is a grid changing function then "FCN1(v1+v2)" will be a
*    child of "A" and "v1+v2" will be a grand-child. The routine INIT_UVAR
*    will have designated "A" as the parent of "v1+v2". The purpose of
*    this routine would be to return the uvar of "A" when given the child_uvar
*    whose definition is "v1+v2"

* V510: *sh* 1/00

* calling argument declarations
	INTEGER		start, child_uvar

* internal variable declarations
	INTEGER   uvar, c, offset

#ifdef unix
	include 'ferret.parm'
	include 'xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* initialize
	uvar = child_uvar
	c = start

	IF (uvar_parent(uvar) .NE. 0) THEN
	   
* compute the character address of this function within its parent
	   READ (uvar_name_code(uvar)(3:5),'(I3)',ERR=5000) offset
	   c = c + offset - 1

	ENDIF

* end of the lineage -- this parent has no parent
	GCF_PARENT_CHAR = c
	RETURN

* funky error exit
 5000	 GCF_PARENT_CHAR = 0	! will generate an error downstream

	   RETURN

	END
