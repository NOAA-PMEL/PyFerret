	CHARACTER*(*) FUNCTION TM_PATH_LIST( path_list, fname, 
     .                               def_ext, do_errmsg, status )

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
* locate the file, filename, with default file extension, def_ext
* (it is acceptable for fname to be blank with the full path given in
*  path_list )

* on unix systems path_list may be a list of directories separated be blanks
* on VMS systems it may be a logical name terminated with a colon

* append default file extension, def_ext, if file has none
* (def_ext is assumed to begin with a ".")

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/9/91
*
* revision 1.0 - 10/17/91 <kob> Relative Version Numbers
*
*  Sun Port 1/30/92 -kob-  added 'ifdef sun' because SUN uses BYTE
*                          where DECstation uses INTEGER*1
*  10/21/92 *sh* changed INQUIRE to OPEN/CLOSE because of incompatibility
*                bug between DEC f772.1 INQUIRE and Multinet VMS/nfs files 
*                (search for "f772.1" to find patches)
* 5/93 *sh* removed the INQUIRE bug fix - now on f77 3.0
* 5/93 *sh* 2 weeks later - restore INQUIRE patch - bug still in f77 ?
*          (PLOT+ fails to find default.spk)
* 4/94 *sh* small change in error message to add " (.des)"
* 11/96 *sh* changed ifdef sun to ifdef inquire_not_nfs_compatible
*	     ... maybe the Muktinet problem has been resolved (??
*	     The use of OPEN instead of INQUIRE fails to allow directories
*	     as valid, existing files. This is a problem for Ferret's
*	     LIST/FORMAT=cache/FILE=directory  (DODS caching)
* 7/97 *kob* Minor modification to error reporting.  Don't want to output
*            a default extension, if an explicit extension is used.
* 12/97 - *kob* - added check for "~/xxx" because it was cashing core dump
*  1/99 - js - on PC's, path separator is ';' rather than ' '
* v5.41 5/02 *acm* - If there are .'s within the filename, the logic for adding
*                    the extension is not adequate (statement 50). When file not
*                    found, go back and see if adding the extension works.
* v5.50 10/02 *acm*- path_buff longer to accomodate longer s_filename 
*                    in descript.def
* v5.42 10/02 *kob* - Increase size of temp_fname to 512
* V6.04  7/07 *acm* - increase size of dataset names to 1024. 
* V701  7/16 *kms*  - increase size of dataset names to 2048
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 12/2022 *acm* Code cleanup remove obsolete ifdefs and f772.1 bug workaround 


* calling argument declarations:
      LOGICAL       do_errmsg
      INTEGER       status
      CHARACTER*(*) path_list, fname, def_ext

**************************** UNIX **********************************

* local variable declarations
      LOGICAL TM_F7721BUG_INQUIRE   ! temporary bug fix for f772.1
      INTEGER slen
      PARAMETER ( slen = 2048 )
      INTEGER TM_LENSTR1, TM_LENSTR
      INTEGER start, end, plen, flen, len_pbuff, postslash, i
      INTEGER tilde, tilde_len, dot,path_len,lext           
      INTEGER vers_int,vers_len,end_tilde,path_tries
      LOGICAL file_exist,pos_ver,zero_ver
      LOGICAL relative, dot_next_to_tilde, retry_ext
      CHARACTER  ext*8, path_buff*(slen),suffix*16,TM_LEFINT*8
      CHARACTER  tilde_buff*8,vers_no*8,temp_fname*(slen)
      INTEGER*1 nhol(slen), vhol(slen),fhol(slen),phol(slen)

      include 'tmap_errors.parm'
#include "gt_lib.parm"

* initialize
      retry_ext = .false.

  60  CONTINUE
*      end = 0          10/91
      end = -1
      path_tries = 0
      plen = LEN( path_list )
      flen = TM_LENSTR1( fname )
      status = merr_erreq
      pos_ver= .false.
      zero_ver = .false.
      dot_next_to_tilde = .false.

* check for relative version number in file name. eg ferret.jnl.~nnn~
* If nnn > or = 0 then relative is false, else it is true.
*
* scan the whole filename and set dot equal to the last period in
* the filename. e.g. for ferret.jnl.~-1~ dot would be 11
      dot = 0
      do 5 i=1,flen
         if (fname(i:i) .EQ. '.')
     .        dot = i
 5    continue
              
* set the tilde buff to be either ".~nnn~" or blank. Also reset filename
* length so that it doesn't include the ".~nnn~"
*
* kob 12/9/97 - "go ~/xxxx" caused a crash, and the reason is that in that 
*                instance, fname is only length 1, so tilde+1 exceeds the 
*                bounds - 
      tilde = INDEX(fname,'~')
      flen = LEN(fname)
      IF (tilde .EQ. flen) THEN
	 end_tilde = INDEX(fname(tilde:), '~')
      ELSE
	 end_tilde = INDEX(fname(tilde+1:), '~')
      ENDIF
      if (dot .EQ. tilde-1) dot_next_to_tilde = .true.

* if tilde eq flen, then this isn't  a relative version issue *kob* 2/98
      if (tilde .EQ. 0 .OR. tilde .EQ. flen) then
         tilde_buff = ' '
      elseif ((tilde .GT. 0) .AND. (dot_next_to_tilde)) THEN
         flen = tilde - 2
         tilde_buff = fname(tilde-1: )
         path_buff = fname(:flen)
      else
         flen = tilde -1
         tilde_buff = fname(tilde: )
         path_buff = fname(:flen)
      endif

* set the relative flag accordingly
      relative = .false.
* if tilde eq flen, then this isn't  a relative version issue *kob* 2/98
      if (tilde .EQ. 0 .OR. tilde .EQ. flen) then
         goto 50
      elseif (fname(tilde+1:tilde+1) .EQ. '0') then
         zero_ver = .true.
      elseif (fname(tilde+1:tilde+1) .NE. '-') then
         pos_ver = .true.
      else
         relative = .true.
      endif   

* apply default extension if filename doesn't have one
* must check if there is a "dot" but ...
* avoid detecting "../path/name" as a significant "dot"

 50      IF ( INDEX( fname(:flen), '.' ) .GT. 0 .AND. 
     .        .NOT. retry_ext)  THEN

         lext = INDEX( fname(:flen), '.' )      ! just to see what this is...
         ext = ' '
         lext= 0
      ELSE
         ext = def_ext
         lext = TM_LENSTR1(ext)
      ENDIF

* does the filename contain its own path ?

      flen = TM_LENSTR1( fname )

      IF ( path_list .EQ. ' ' ) THEN
         IF ( fname .EQ. ' ' ) GOTO 1000   ! INQUIRE bug: blank name exists
         path_buff = fname(:flen)//ext
         end = plen    ! flag not to continue search after one try
         GOTO 200
      ENDIF

* find start of next path text (non-blank)
 100  IF ( end .ge. plen - 2 ) GOTO 1000   ! possible path list truncation
      DO start = end+2, plen
        IF ( path_list(start:start) .NE. EnvSeparator ) GOTO 110
      ENDDO

* no more paths
      GOTO 1000

* find end of path name (last non-blank) (note: mandatory blank at end)
 110  end = INDEX(path_list(start+1:plen), EnvSeparator) + start - 1
      if (end .lt. start) then
         end = plen-1
      endif

* check path name and file name
* note: FORTRAN/RISC compiler doesn't permit // with character*(*)
*      (yet it is working w/ fname(:flen) ... perhaps it requires explicit len
      IF ( fname .EQ. ' ' ) THEN
* ... filename is contained in the path list environment variable
         path_buff = path_list(start:end)
      ELSEIF ( pos_ver) then
            IF (dot_next_to_tilde) then
                path_buff = path_list(start:end)//PathSeparator
     .           //fname(:flen)//ext(:lext)//tilde_buff
            ELSE
                 path_buff = path_list(start:end)//PathSeparator
     .              //fname(:flen)//ext(:lext)//'.'//tilde_buff
            ENDIF
            IF (end_tilde .EQ. 0) THEN
               len_pbuff = TM_LENSTR1( path_buff )
               path_buff = path_buff(:len_pbuff)//'~'
            ENDIF
      ELSE
         path_buff = path_list(start:end)//PathSeparator
     .              //fname(:flen)//ext
      ENDIF

* for relative versions, we need to convert the tilde_buff, filename
* and path list to c strings and pass them in to tm_make_relative_ver
* which will return the proper version number for the file to open
 200  if ( relative) then
            temp_fname = fname(:flen)//ext
            path_len = TM_LENSTR1(temp_fname)
            tilde_len = TM_LENSTR1( tilde_buff )
            CALL TM_FTOC_STRNG( tilde_buff(:tilde_len), nhol, slen)
            CALL TM_FTOC_STRNG(temp_fname(:path_len) , fhol, slen)
            CALL TM_FTOC_STRNG(path_list(start:end),phol,slen)
            call TM_MAKE_RELATIVE_VER(nhol, fhol,phol,vers_int)
*            call TM_CTOF_STRNG( vhol, vers_no, slen)
            vers_no =  TM_LEFINT(vers_int,vers_len)
            path_len = TM_LENSTR1(path_buff)
            path_buff = path_buff(:path_len)//'.'//'~'
     .                      //vers_no(:vers_len)//'~'
         endif
      len_pbuff = TM_LENSTR1( path_buff )

      INQUIRE ( FILE		= path_buff( :len_pbuff ) ,
     .			EXIST		= file_exist,
     .			ERR 		= 5000)

      IF ( file_exist ) GOTO 1100

* try next path in list
      path_tries = path_tries + 1
      GOTO 100

* file not found
 1000 CONTINUE

* try forcing addition of the default extension -- if there are internal .'s 
* in the filename, the IF test at Statement# 50 is not adequate.  Go back, 
* reinitialize everything, and try again.

      IF (.NOT. retry_ext) THEN
         ext = def_ext
         lext = TM_LENSTR1(ext)
         retry_ext = .TRUE.
         GO TO 60
      ENDIF

* file still not found

      TM_PATH_LIST = ' '
      status = merr_notmounted
      IF ( do_errmsg ) THEN
         GOTO 5000
      ELSE
         RETURN
      ENDIF

* successful exit
* set TM_PATH_LIST to path_buff because v3.0 of f77 won't let this happen in INQUIRE -10/8/91
 1100 status = merr_ok
      TM_PATH_LIST = path_buff( :len_pbuff)
      RETURN

* error exits
* 7/97 *kob* mods to not include a default extension in error message if
*            an explicit extension is used
 5000 lext = TM_LENSTR(def_ext)      !can now be zero 7/97
      if (path_list(1:1) .EQ. '.' .OR. path_tries .GT. 1) then
         path_buff = fname(:TM_LENSTR1(fname))
	 if ( lext .GT. 0 ) path_buff = path_buff(:TM_LENSTR1(path_buff))
     .				//' ('//def_ext(1:lext)//')' ! *sh* 4/94,11/94
      else 
         path_buff = path_list(:plen-1)//PathSeparator//fname
      endif
      len_pbuff = TM_LENSTR1(path_buff) ! *sh* 4/94
      CALL TM_ERRMSG (status, status, 'TM_PATH_LIST',
     .                unspecified_int4, no_stepfile,
     .                path_buff(:len_pbuff), 
     .                no_errstring, *9900)
 9900 RETURN

      END
