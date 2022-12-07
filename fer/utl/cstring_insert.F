	SUBROUTINE CSTRING_INSERT( fstring, flen, cstring, clen, insert_pt)

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
* Insert a FORTRAN character string into a "C" string (a character array)
* using a carriage return followed by null as the terminator.
* Return with the insertion point pointing to the null.

* V400:  8/94

* Note that insert_pt is both input and output - it will be returned at the
* correct position for the next insert

* If insert_pt>clen upon return that means the buffer has overflowed
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
* calling argument declarations:
	INTEGER	flen, clen, insert_pt
	CHARACTER fstring*(*)
        INTEGER*1 cstring(clen) ! c-type Hollerith string buffer

* internal variable declarations:

* internal parameter declarations:
        INTEGER*1 CR, NULL, STAR
	PARAMETER ( CR   = 13,
     .		    NULL =  0,
     .		    STAR = 42 )	

	IF ( insert_pt + flen + 2 .LE. clen ) THEN

* insert the new string
	  CALL TM_FTOC_STRNG( fstring(:flen), cstring(insert_pt), clen )
	  insert_pt = insert_pt + flen

* insert the terminators
	  cstring(insert_pt) = CR
	  insert_pt = insert_pt + 1
	  cstring(insert_pt) = NULL      ! overwritten by next
	
	ELSE

* ERROR: insufficient room to append this text
	  DO insert_pt = insert_pt, clen-1
	     cstring(insert_pt) = STAR
 	  ENDDO
	  cstring(clen) = NULL
	  insert_pt = clen + 1

	ENDIF

	RETURN
	END
