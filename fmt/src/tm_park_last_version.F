	SUBROUTINE TM_PARK_LAST_VERSION ( fname, status )

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

* calling argument declarations:
      INTEGER       status
      CHARACTER*(*) fname

#ifdef unix
      include 'tmap_format/tmap_errors.parm'
#ifdef AIX_XLF
      include 'tmap_format/xrisc_buff.cmn'
#endif

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
      IF ( INDEX( fname, '/') .GT. 0) THEN
         flen = TM_LENSTR1(fname)
         do 10 i = flen-1,1,-1
            IF ( fname(i:i) .EQ. '/' ) THEN
               postslash = MIN(i+1,flen)     ! MIN if slash at end->dont crash
               GOTO 11
            ENDIF
 10      continue
* ... next version with path name
 11      CALL TM_NEXT_VER_NAME( fname(postslash:), buff,
     .			        fname(:postslash-1))
#ifdef AIX_XLF
  	 risc_buff = fname(:postslash-1)
         CALL TM_RENAME( fname, risc_buff//buff, status )
#else
         CALL TM_RENAME( fname, fname(:postslash-1)//buff, status )
#endif
      ELSE
* ... next version in this directory
         CALL TM_NEXT_VER_NAME(fname,buff,' ')
         CALL TM_RENAME( fname, buff, status )
      ENDIF

      RETURN
#else
      STOP 'TM_PARK_LAST_VERSION is for Unix systems, only'
#endif
      END

