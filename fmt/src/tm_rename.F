	SUBROUTINE TM_RENAME ( oldname, newname, status )

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
* rename a file

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix and VMS operating systems
*
* rev 0.0 -   5/6/91

* calling argument declarations:
        INTEGER  status
        CHARACTER*(*) oldname, newname

* internal variable declarations:
	INTEGER	olen, nlen
        INTEGER TM_LENSTR1

#ifdef unix
        include 'tmap_format/tmap_errors.parm'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

#ifdef unix
        INTEGER slen
        PARAMETER  ( slen = 128 )
        INTEGER oldhol(slen), newhol(slen)  ! c-type Hollerith string buffers

* initialize
        olen = TM_LENSTR1(oldname)
        nlen = TM_LENSTR1(newname)

* translate to null-terminated c-type strings
        CALL TM_FTOC_STRNG( oldname(:olen), oldhol, slen )
        CALL TM_FTOC_STRNG( newname(:nlen), newhol, slen )

* call c-written routine to make system call
        CALL TM_C_RENAME( oldhol, newhol, status )

* translate Unix system error code to TMAP library form
        IF ( status .EQ. 0 ) THEN
           status = merr_ok
        ELSE
           status = merr_erreq
        ENDIF

#else
        STOP 'TM_RENAME for VMS isn't yet written'
!    ... insert VMS system call here

#endif
	RETURN
	END

