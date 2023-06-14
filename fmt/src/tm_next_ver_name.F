	SUBROUTINE TM_NEXT_VER_NAME ( name, vernam, path )

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
* determine the name for the next version of this file
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix and VMS operating systems
*
* rev 0.0 -   5/6/91
* rev 1.0 -   10/17/91  -- <kob>
*                      - modified to pass the specified path to tm_c_ver_name
*                        to allow searching for files in directories other
*                        than just the current one.
*  Sun Port 1/30/92 -kob-  added 'ifdef sun' because SUN uses BYTE where DECstation uses
*                          INTEGER*1
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* calling argument declarations:
        CHARACTER*(*) name, vernam, path

* internal variable declarations:
	INTEGER	nlen
        INTEGER TM_LENSTR1

        include 'tmap_errors.parm'

        INTEGER slen,plen
        PARAMETER  ( slen = 128 )
        INTEGER*1 nhol(slen), vhol(slen),phol(slen) ! c-type Hollerith string buffers

* initialize
        nlen = TM_LENSTR1(name)
        plen = TM_LENSTR1(path)

* translate to null-terminated c-type strings
        CALL TM_FTOC_STRNG( name  (:nlen), nhol, slen )
        CALL TM_FTOC_STRNG( path  (:plen), phol, slen)

* call c-written routine to determine next version name from directory
* pass in path so that search is not limited to current directory 10/17/91
        CALL TM_C_VER_NAME( nhol, vhol, phol )

* translate new version name to FORTRAN style
        CALL TM_CTOF_STRNG( vhol, vernam, slen )

	RETURN
	END

