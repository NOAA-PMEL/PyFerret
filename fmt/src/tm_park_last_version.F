	SUBROUTINE TM_PARK_LAST_VERSION ( fname, status )

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
* rename any existing file of name fname to the current highest version of
* that filename plus one

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix operating system
*
* rev 0.0 -   5/6/91
* rev 1.0 -   10/17/91  -- <kob>
*                        - extract path out of filename if it exists as part of
*                           the filename. pass both filename and path to 
*                           tm_next_ver_name.
*SUN port - 2/92- kob-  ***Temporarily***
*                         don't call tm_next_ver_name or tm_rename if running
*                         on  a sun.  Merely append '.old' the the file
*                         if it exists.  
* 		comment out ifdef because posix version should work 3/92 kob
* bug fix 1/94 *sh* - if path is included with file name then pass only the
*			name as the first argument to TM_NEXT_VER_NAME
*			Re-append path to the version name.
*
* IBM port 6/2/94 *kob* - Inherited length string concatenation error on IBM
*			  aix compiler forces uses of risc_buff
*
* 1/03 - *kob* g77 port - Need to use risc_buff for arbitrary concatenations

* calling argument declarations:
      INTEGER       status
      CHARACTER*(*) fname

      include 'tmap_errors.parm'
#include "gt_lib.parm"
      include 'xrisc_buff.cmn'

* internal variable declarations:
      LOGICAL there
      CHARACTER*128  buff
      INTEGER flen, TM_LENSTR1, i
      INTEGER postslash
* Initialize
      status = merr_ok

* does the file already exist ?
      INQUIRE ( FILE=fname, EXIST=there )

      IF ( .NOT.there ) RETURN

* it already exists - rename it to the highest version
* first extracting possible path out of filename  -- 10/91
      IF ( INDEX( fname, PathSeparator) .GT. 0) THEN
         flen = TM_LENSTR1(fname)
         do 10 i = flen-1,1,-1
            IF ( fname(i:i) .EQ. PathSeparator ) THEN
               postslash = MIN(i+1,flen)     ! MIN if slash at end->dont crash
               GOTO 11
            ENDIF
 10      continue
* ... next version with path name
 11      CALL TM_NEXT_VER_NAME( fname(postslash:), buff,
     .			        fname(:postslash-1))
  	 risc_buff = fname(:postslash-1)
         CALL TM_RENAME( fname, risc_buff(:TM_LENSTR1(risc_buff))//buff, 
     .                   status )
      ELSE
* ... next version in this directory
         CALL TM_NEXT_VER_NAME(fname,buff,' ')
         CALL TM_RENAME( fname, buff, status )
      ENDIF

      RETURN
      END

