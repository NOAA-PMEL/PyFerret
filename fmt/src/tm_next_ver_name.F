	SUBROUTINE TM_NEXT_VER_NAME ( name, vernam, path )

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

* calling argument declarations:
        CHARACTER*(*) name, vernam, path

* internal variable declarations:
	INTEGER	nlen
        INTEGER TM_LENSTR1

#ifdef unix
        include 'tmap_format/tmap_errors.parm'
#endif

#ifdef unix
        INTEGER slen,plen
        PARAMETER  ( slen = 128 )
#ifdef sun
        BYTE      nhol(slen), vhol(slen),phol(slen) ! c-type Hollerith string buffers
#else
        INTEGER*1 nhol(slen), vhol(slen),phol(slen) ! c-type Hollerith string buffers
#endif

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

#else
        STOP 'TM_NEXT_VER_NAME isn't used by VMS'
#endif
	RETURN
	END

