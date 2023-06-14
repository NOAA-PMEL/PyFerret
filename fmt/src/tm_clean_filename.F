	CHARACTER*(*) FUNCTION TM_CLEAN_FILENAME ( pathname )

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
* remove the directory path and the extension from a full pathname leaving
* only the filename

*  written 6/92 by S.H. for PMEL/TMAP
* 5/93 - *sh* modified so endings other than .cdf and .des are returned
* v5.32 *kob* - CYGWIN Prob - when running under Win32/Cygwin, have to
*  3/01         to check to see if file from which a clean filename is
*               being extracted is a DODS file.  If it is, we have to use
*               the "typical" unix path separator of "/" rather than the
*               CYGWIN path separator of "\".  Otherwise we don't get a 
*               clean filename extracted from the dataset.  This will 
*               cause problems in find_dset_number and find_dset_name
* 2/03 - *acm*  clean off .nc as well as .cdf and .des
*               Note this may change the names of dynamic grids, which are
*               generated from dataset names
* 10/18 - *acm* Call TM_LENSTR1 not the LEN fucnction. Remove old cygwin stuff.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

#include "gt_lib.parm"

* argument
	CHARACTER*(*) pathname

* local variable
        INTEGER TM_LENSTR1, idot, islash, plen

* initialize
        plen = TM_LENSTR1( pathname )

* locate the last dot in the name
        DO idot = plen, 1, -1
           IF ( pathname(idot:idot) .EQ. '.' ) GOTO 190
        ENDDO
	idot = plen + 1    ! no dot
 190    idot = idot - 1

* 5/93 other than .des or .cdf should not be cleaned off 
	IF ( plen-idot .EQ. 4 ) THEN
	   IF (   pathname(idot+1:plen) .NE. '.cdf'
     .      .AND. pathname(idot+1:plen) .NE. '.des' ) idot = plen
	ELSEIF ( plen-idot .EQ. 3 ) THEN
	   IF (   pathname(idot+1:plen) .NE. '.nc' ) idot = plen
	ELSE
	   idot = plen
	ENDIF

* locate the last slash in the name

        DO islash = plen, 1, -1
           IF ( pathname(islash:islash) .EQ. PathSeparator ) GOTO 290
        ENDDO
        islash = 0    ! no slash
 290    islash = islash + 1

* check for pathological name
        IF ( islash .GT. idot ) idot = plen
	
        TM_CLEAN_FILENAME = pathname(islash:idot)

	RETURN
	END

