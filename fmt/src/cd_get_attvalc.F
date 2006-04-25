	LOGICAL FUNCTION CD_GET_ATTVALC ( dset, varid, attrib,
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
* 1/03 **ACM** add this note: (no code changes)
*      Note that under OSF (and PC? ) CD_GET_ATTVALC returns TRUE when 
*      the value of the attribute contains a non-numeric string, say "F".  
*      Under Solaris and Linux, returns FALSE. If the value of the modulo 
*      attribute is type character, this function tries to read a numeric 
*      value from a character buffer . For solaris, linux, this read:
*
*      	     READ (buff,*,ERR=5000) val  ! convert text to value
*
*      returns an error and goes to 5000 if the contents of buff is not  
*      a number.  For OSF, reads anyway, returning -1 when the character 
*      string is T, or 0 for all other characters. 
* V600 *ACM* attribute control. 
*             Replace NCAINQ with CD_GET_VAR_ATT_INFO
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib from linked list structure
*             CD_GET_ATTVAL replaced by NC_GET_ATTRIB, which now returns  
*               strings or real values according to attrib type


* argument definitions:
*	dset	 - dataset id number for already opened CDF file
*       varid    - variable id in netCDF file
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*       vname    - name of variable owning attribute (used in error msgs)
*       val      - array into which to return result
*       blank_val- if a blank character attribute, then return this default

* argument declarations
      LOGICAL       do_warn
      INTEGER	    dset, varid
      CHARACTER*(*) attrib, vname
      REAL	    val, blank_val

      include 'xio.parm'
      include 'netcdf.decl'
#include "netcdf.inc"
      include 'tmap_errors.parm'

* internal variable declarations:
      LOGICAL NC_GET_ATTRIB
      INTEGER attlen, attoutflag
      INTEGER TM_LENSTR1, lena, attype, status
      REAL tmp
      CHARACTER buff*132

* inquire about the attribute by name
      lena = TM_LENSTR1( attrib )
      CALL CD_GET_VAR_ATT_INFO (dset, varid, attrib(:lena), attype, 
     .     attlen, attoutflag, status )

      IF ( status .EQ. merr_ok ) THEN
	IF (attype .EQ. NCCHAR) THEN
* get a text attribute as a value

	   CD_GET_ATTVALC = NC_GET_ATTRIB(dset,varid,attrib(:lena),
     .                            do_warn,vname,132, attlen,  
     .                            attoutflag, buff, tmp )
	   IF (.NOT.CD_GET_ATTVALC) THEN
	     RETURN
	   ELSEIF (buff .EQ. ' ') THEN
	     val = blank_val
	   ELSE
	     READ (buff,*,ERR=5000) val  ! convert text to value
	   ENDIF 
	 ELSE
* get a value attribute
	   CD_GET_ATTVALC = NC_GET_ATTRIB(dset, varid, attrib(:lena),
     .                                    do_warn, vname, 1, attlen,
     .                                    attoutflag,  buff, val )
	 ENDIF
      ELSE
* didnt get nothin
	 CD_GET_ATTVALC = .FALSE.
      ENDIF

      RETURN

* error exit
 5000 CONTINUE
c      IF (do_warn) THEN
c	CALL TM_NOTE('Undecipherable value of netCDF attribute '//
c     .		      attrib(:lena)//' on variable '//vname, tooplu)
c	CALL TM_NOTE('"buff(:TM_LENSTR1(buff))"', tooplu)
c      ENDIF
      CD_GET_ATTVALC = .FALSE.
      RETURN
      END
