	CHARACTER*(*) FUNCTION TM_PATH_LIST( path_list, fname, def_ext,
     .                                       do_errmsg, status )

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
* calling argument declarations:
      LOGICAL       do_errmsg
      INTEGER       status
      CHARACTER*(*) path_list, fname, def_ext

#ifdef unix
**************************** UNIX **********************************

* local variable declarations
      LOGICAL TM_F7721BUG_INQUIRE   ! temporary bug fix for f772.1
      INTEGER slen
      PARAMETER ( slen = 64)
      INTEGER TM_LENSTR1, TM_LENSTR
      INTEGER start, end, plen, flen, len_pbuff, postslash, i
      INTEGER tilde, tilde_len, dot,path_len,lext           
      INTEGER vers_int,vers_len,end_tilde,path_tries
      LOGICAL file_exist,pos_ver,zero_ver
      LOGICAL relative, dot_next_to_tilde
      CHARACTER  ext*8, path_buff*128,suffix*16,TM_LEFINT*8
      CHARACTER  tilde_buff*8,vers_no*8,temp_fname*128
*
#ifdef sun
      BYTE      nhol(slen), vhol(slen),fhol(slen),phol(slen)
                                !c-type Hollerith string buffers
#else
      INTEGER*1 nhol(slen), vhol(slen),fhol(slen),phol(slen)
                                !c-type Hollerith string buffers
#endif
*
      include 'tmap_format/tmap_errors.parm'
      include 'tmap_format/gt_lib.parm'

* initialize
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
 5              continue
              
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
 50      IF ( INDEX( fname(:flen), '.' ) .GT. 0 ) THEN
         ext = ' '
         lext= 0
      ELSE
         ext = def_ext
         lext = TM_LENSTR1(ext)
      ENDIF


* does the filename contain its own path ?
      IF ( path_list .EQ. ' ' ) THEN
         IF ( fname .EQ. ' ' ) GOTO 1000   ! INQUIRE bug: blank name exists
         path_buff = fname(:flen)//ext
         end = plen    ! flag not to continue search after one try
         GOTO 200
      ENDIF

* find start of next path text (non-blank)
 100  IF ( end .ge. plen - 2 ) GOTO 1000   ! possible path list truncation
*      DO 101 start = end+1, plen            10/91
      DO 101 start = end+2, plen
 101  IF ( path_list(start:start) .NE. ' ' ) GOTO 110
* no more paths
      GOTO 1000

* find end of path name (last non-blank) (note: mandatory blank at end)
 110  end = INDEX(path_list(start+1:plen), ' ') + start - 1

* check path name and file name
* note: FORTRAN/RISC compiler doesn't permit // with character*(*)
*      (yet it is working w/ fname(:flen) ... perhaps it requires explicit len
      IF ( fname .EQ. ' ' ) THEN
* ... filename is contained in the path list environment variable
         path_buff = path_list(start:end)
      ELSEIF ( pos_ver) then
            IF (dot_next_to_tilde) then
                path_buff = path_list(start:end)//'/'//fname(:flen)
     .           //ext(:lext)//tilde_buff
            ELSE
                 path_buff = path_list(start:end)//'/'//fname(:flen)
     .              //ext(:lext)//'.'//tilde_buff
            ENDIF
            IF (end_tilde .EQ. 0) THEN
               len_pbuff = TM_LENSTR1( path_buff )
               path_buff = path_buff(:len_pbuff)//'~'
            ENDIF
      ELSE
         path_buff = path_list(start:end)//'/'//fname(:flen)//ext
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
* INQUIRE bug check was added for f772.1 problem.
* the INQUIRE check was commented out 5/13/93 for f77 3.2
* but was restored May 28 cuz the bug was still there
* on June 9 this comment was removed from the ifdef line
* On 11/96 the ifdef sun was changed to ifdef inquire_not_nfs_compatible
* (and the ifdef logic was reversed)
#ifdef inquire_not_nfs_compatible
! temporary routine needed because of incompatibility between DEC f772.1 and
! Multinet nfs-mounted files (files sometimes not found)  *sh* 10/21/92
      file_exist = TM_F7721BUG_INQUIRE( path_buff( :len_pbuff ) )
#else
      INQUIRE (	        FILE		= path_buff( :len_pbuff ) ,
     .			EXIST		= file_exist,
     .			ERR 		= 5000)
* took out NAME statement because v3.0 of f77 didn't dig it ------kob 10/8/91
*     .		   	NAME		= TM_PATH_LIST,
#endif
      IF ( file_exist ) GOTO 1100

* try next path in list
      path_tries = path_tries + 1
      GOTO 100

* file not found
 1000 TM_PATH_LIST = ' '
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
         path_buff = path_list(:plen-1)//'/'//fname
      endif
      len_pbuff = TM_LENSTR1(path_buff) ! *sh* 4/94
      CALL TM_ERRMSG (status, status, 'TM_PATH_LIST',
     .                unspecified_int4, no_stepfile,
     .                path_buff(:len_pbuff), 
     .                no_errstring, *9900)
 9900 RETURN


#else
**************************** VMS **********************************

      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
      INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'

* local variable declarations
      LOGICAL file_exist
      INTEGER TM_LENSTR1
      INTEGER plen, flen

* initialize
      plen = TM_LENSTR1( path_list )
      flen = TM_LENSTR1( fname )
      status = merr_erreq

      INQUIRE ( FILE	        = fname,
     .		DEFAULTFILE	= path_list(:plen)//def_ext,
     .		NAME		= TM_PATH_LIST,
     .          EXIST           = file_exist,
     .		ERR 		= 5000)

      IF ( file_exist ) THEN
	   status = merr_ok
      ELSE
	   status = merr_notmounted
           TM_PATH_LIST = ' '
           IF (do_errmsg) GOTO 5000
      ENDIF	   

* done. 
      RETURN

* error exits
 5000 CALL TM_ERRMSG (status, status, 'TM_PATH_LIST',
     .			unspecified_int4, no_stepfile,
     .			path_list(:plen)//fname(:flen)//def_ext,
     .                  no_errstring, *9900)
 9900 RETURN

#endif
     
      END
