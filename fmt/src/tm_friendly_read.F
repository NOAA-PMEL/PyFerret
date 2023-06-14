	SUBROUTINE TM_FRIENDLY_READ( prompt, buff, * )

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
* get a line of input allowing full gnu-readline line editing
* take abnormal return if eof is encountered

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix and VMS operating systems
*
* rev 0.0 -   5/9/91
*  Sun Port 1/30/92 -kob-  added ifdef sun because uses BYTE whereas
*  DECstation uses  INTEGER*1
* 1/94 *sh* increased max read buffer len (blen) from 256 to 512
* 4/01 *sh* increased blen again -- to 2048

* calling argument declarations:
        CHARACTER*(*) prompt, buff

* internal variable declarations:
! "blen" must exceed the longest read
        INTEGER plen, blen
        PARAMETER (plen = 8, blen = 2048 )

        INTEGER*1 hprompt(plen), hbuff(blen)

* translate to null-terminated c-type strings
        CALL TM_FTOC_STRNG( prompt, hprompt, plen )

* call c-written routine to get input line
        CALL TM_FTOC_READLINE( hprompt, hbuff )

* translate received input line to FORTRAN character array
        CALL TM_CTOF_STRNG( hbuff, buff, blen )

* null string returned means that EOF (^D) was entered
        IF ( hbuff(1) .EQ. 4 ) RETURN 1     ! ^D passed

	RETURN
	END

