      CHARACTER*(*)FUNCTION TM_INQ_PATH
     .   ( fname, path, def_ext, do_errmsg, status )
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
*
* Find the full path name of the given file
* failure to find a file in a status of merr_not_mounted
* depending on the logical do_errmsg an error message may be issued

* using this routine is "nearly" OS-independent between VMS and UNIX
* environment variables substitute for VMS logicals
* the interpretation of the default extension is a little different
* ... on UNIX it must be ".xxx" on VMS it can be name.xxx

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS

* revision 0.00 - 10/23/89 - code lifted from TM_INIT_DSET by Mark Verschell
* Unix/RISC FORTRAN port 3/7/91 *sh* - OS-specific file syntaxes
*                          - generalized to promote VMS/Unix similarities
* revision 1.00 - 10/17/91 - <kob>
*                          - separate path (if it exists) from file name
*                            and pass them to tm_path_list as two unique
*                            variables.
* revison 1.01  - 2/13/92  - increased size of TM_PATH_LIST, path_list to 512
*  *sh* 2/12/93 - corrected bug due to increased path_list (2/13/92) - it was
*		  equivalenced to "risc_buff" which was too short
*
*  *sh* 5/28/93 - if filename begins with $ then translate the environment var
*		  (this is a partial sol'n - doesn't deal with embedded $'s)
*
* IBM port 14apr94 *kob* : IBM will only allow character string lengths
*			    between 1 and 500.
*			 : CHARACTER*(*) variables cannot be used in concatenations,
*			   so I used risc_buff.
* v530 *kob* 1/00 - minor mods for remote mc aggregations
* 1/03 - *kob* g77 port - Need to use risc_buff for arbitrary concatenations.  Remove
*                         ifdef AIX and just use risc_buff by default
* 2/05 *acm* V58+  increase length of path_list string for FDS and other cases
*                  where environment variables get very long (bug 1169)
* V685+ 12/13    - *acm*  bounds checking: dont refer to parts of the string outside its bounds.

* arguments:
*	fname	   - filename - possibly lacking extension
*                             - possibly including explicit path 
*       path       - default logical or environment variable with path
*	def_ext	   - default file extension
*       do_errmsg  - give error message for file-not-found

* Argument definitions
        LOGICAL         do_errmsg
	INTEGER		set_cnt, status
	CHARACTER*(*)	fname, path, def_ext

* Parameter and common files
	include 'tmap_errors.parm'		! error codes
#include "gt_lib.parm"
	include 'xrisc_buff.cmn'


* local valiable declarations
        CHARACTER TM_PATH_LIST*8192, path_list*8192
        INTEGER TM_LENSTR1, STR_SAME, plen,flen,i,postslash
	LOGICAL is_http

* path may be an environment variable (UNIX) with a path list or
* a logical name (VMS).  Failing that, it is ignored.
*

* reset postslash to 0 every time code enters here. If not reset, it causes
* a problem with the relative version numbers, cutting off first character
* of filename.
      postslash = 0
      flen = TM_LENSTR1(fname)

* Set up a logical. With strict bounds checking, a string of length 3
* cant be used in  *** IF ( fname(1:4) .EQ. 'http') THEN ***

       is_http = .FALSE.
       IF (flen .GT. 3) THEN
          is_http  = (fname(1:4) .EQ. 'http')
       ENDIF

* if a pathname exists, we want to extract it from the file name,  and
* pass them to tm_path_list as two separate, but lovely, entities.
* *** start of 5/93 addition
      IF ( fname(1:1) .EQ. '$' ) THEN
* ... translate "$" environment variable at beginning of filename
	 postslash = INDEX( fname ,PathSeparator ) + 1
	 IF ( postslash .EQ. 1 ) THEN
	    CALL GETENV( fname(2:flen), path_list )
	    postslash = flen + 1       ! entire filename from environment var
	 ELSE
	    CALL GETENV( fname(2:postslash-2), path_list )
	 ENDIF		
	 IF ( path_list .EQ. ' ' ) GOTO 5202
	 plen = TM_LENSTR1( path_list )
* ... transfer "morepath" info in the case of $ENVIRONMENT/morepath/filename
	 DO 5 i = flen-1,postslash+1,-1
	    IF ( fname(i:i) .EQ. PathSeparator ) THEN
	       path_list = path_list(:plen)//PathSeparator
     .              //fname(postslash:i-1)
	       plen = TM_LENSTR1( path_list )
	       postslash = i+1
	       GOTO 6
	    ENDIF
 5	 CONTINUE
 6	 IF ( plen .LT. LEN(path_list) ) THEN
            plen = plen + 1                ! mandatory ' ' at end
         ELSE
            GOTO 5100
         ENDIF
* *** end of 5/93 addition
      ELSEIF ( path .EQ. ' ' ) THEN
         path_list = ' '
         plen = 1
* *kob* 11/00 Allow DODS files as MC sets - just use full http address
      ELSEIF ( is_http ) THEN
         TM_INQ_PATH=fname(:flen)
         status = merr_ok
         GOTO 9900
      ELSEIF ( INDEX( fname ,PathSeparator ) .GT. 0 ) THEN
         DO 10 i = flen-1,1,-1
            IF ( fname(i:i) .EQ. PathSeparator ) THEN
               postslash = i+1
               GOTO 11
         ENDIF
 10      continue
 11      path_list =   fname(:postslash-2)
         plen = TM_LENSTR1(path_list)
         plen = plen + 1
* Otherwise, use the environment paths.
      ELSE
         CALL GETENV( path, path_list )    ! ' ' if no translation
         plen = TM_LENSTR1( path_list )
         IF ( plen .LT. LEN(path_list) ) THEN
            plen = plen + 1                ! mandatory ' ' at end
         ELSE
            GOTO 5100
         ENDIF
      ENDIF

* evaluate the path
* Send path and filename seperately to tm_path_list.
      IF (postslash .EQ. 0) postslash = 1

! The INQUIRE statement inside TM_PATH_LIST fails on names longer 
! than 256 chars causing a BUS error, so dont look in the DODS 
! cache for the file.
c
c is the above still true?
c
      flen = TM_LENSTR1(fname)
c      IF (flen .GT. 256 .AND.
c     .  STR_SAME(path, 'FER_DODS_CACHE_DIR').EQ.0 ) THEN 
c         status = merr_notmounted
c
c      ELSE
         IF (postslash .LE. flen) THEN
            TM_INQ_PATH = TM_PATH_LIST( path_list(:plen), 
     .                               fname(postslash:),def_ext, 
     .                               do_errmsg, status )
         ELSE   ! this option if entire name is an environment var
            TM_INQ_PATH = TM_PATH_LIST( path_list(:plen), 
     .                               ' ',def_ext, 
     .                               do_errmsg, status )
         ENDIF
c      ENDIF

      RETURN

* error exits
 5100 CALL TM_ERRMSG (merr_syntax, status, 'TM_INQ_PATH',
     .			unspecified_int4, no_stepfile,
     .			'path list too long',
     .                  no_errstring, *9900)
 5202 risc_buff = fname
      CALL TM_ERRMSG (merr_syntax, status, 'TM_INQ_PATH',
     .			unspecified_int4, no_stepfile,
     .			'unknown environment variable: '//risc_buff(1:flen),
     .                  no_errstring, *9900)
 9900 RETURN
      END

