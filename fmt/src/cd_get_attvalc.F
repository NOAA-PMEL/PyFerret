	LOGICAL FUNCTION CD_GET_ATTVALC ( cdfid, varid, attrib,
     .                                    do_warn, vname,
     .                                    val, blank_val )

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
* This routine is like cd_attval except that it allows for the possibility
* that the required numerical value is stored in a character attribute
* and it can return only a single value

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* 2/02  *sh*

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*       vname    - name of variable owning attribute (used in error msgs)
*       val      - array into which to return result
*       blank_val- if a blank character attribute, then return this default

* argument declarations
      LOGICAL       do_warn
      INTEGER	    cdfid, varid
      CHARACTER*(*) attrib, vname
      REAL	    val, blank_val

      include 'xio.parm'
      include 'netcdf.decl'
#include "netcdf.inc"

* internal variable declarations:
      LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL
      INTEGER vback
      INTEGER TM_LENSTR1, lena, attype, cdfstat
      CHARACTER buff*132


* inquire about the attribute by name
      lena = TM_LENSTR1( attrib )
      CALL NCAINQ(cdfid, varid, attrib(:lena), attype, vback, cdfstat )

      IF ( cdfstat .EQ. NCNOERR ) THEN
	IF (attype .EQ. NCCHAR) THEN
* get a text attribute as a value
	   CD_GET_ATTVALC = CD_GET_ATTRIB(cdfid,varid,attrib(:lena),
     .                                    do_warn,vname,buff,132 )
	   IF (.NOT.CD_GET_ATTVALC) THEN
	     RETURN
	   ELSEIF (buff .EQ. ' ') THEN
	     val = blank_val
	   ELSE
	     READ (buff,*,ERR=5000) val  ! convert text to value
	   ENDIF 
	 ELSE
* get a value attribute
	   CD_GET_ATTVALC = CD_GET_ATTVAL(cdfid,varid,attrib(:lena),
     .                                    do_warn,vname,val,1,vback )
	 ENDIF
      ELSE
* didnt get nothin
	 CD_GET_ATTVALC = .FALSE.
      ENDIF

      RETURN

* error exit
 5000 IF (do_warn) THEN
	CALL TM_NOTE('Undecipherable value of netCDF attribute '//
     .		      attrib(:lena)//' on variable '//vname, tooplu)
	CALL TM_NOTE('"buff(:TM_LENSTR1(buff))"', tooplu)
      ENDIF
      CD_GET_ATTVALC = .FALSE.
      RETURN
      END
