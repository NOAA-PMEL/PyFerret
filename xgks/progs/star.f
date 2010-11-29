C      PROGRAM STAR
C		Copyright IBM Corporation 1989
C
C                      All Rights Reserved
C
C Permission to use, copy, modify, and distribute this software and its
C documentation for any purpose and without fee is hereby granted,
C provided that the above copyright notice appear in all copies and that
C both that copyright notice and this permission notice appear in
C supporting documentation, and that the name of IBM not be
C used in advertising or publicity pertaining to distribution of the
C software without specific, written prior permission.
C
C IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
C ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
C IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
C ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
C WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
C ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
C SOFTWARE.
C
C
C      DESCRIPTION:
C       This program draws a yellow star on a blue background and writes
C       the title 'Star' in green in under the star.
C      CONFORMANCE:
C       GKS level: ma
C       FORTRAN-77 binding with FORTRAN-77 Subset comments.
C       At least one output or outin workstation.
C
C      Define GKS constants.
C
       INTEGER GSOLID
       PARAMETER ( GSOLID = 1 )
       INTEGER GACENT,GAHALF
       PARAMETER ( GACENT = 2, GAHALF = 3 )
C
C      Implementation dependent constants.
C
       INTEGER ERROUT,TTOUT,WSTYPE,NBYTES
       PARAMETER ( ERROUT = 1, TTOUT = 5, WSTYPE = 4, NBYTES = -1 )
C
C      Define coordinates for drawing the star.
C   
       REAL STARX( 5 ),STARY( 5 )
       DATA STARX / 0.951057, -0.951057, 0.587785, 0.0, -0.587785 /
       DATA STARY / 0.309017, 0.309017, -0.951057, 1.0, -0.951057 /
C
C      Perform implementation dependent initialization
C
CC     OPEN ( TTOUT,STATUS='NEW' )
C      Open GKS and activate a workstation.
C
       CALL GOPKS ( ERROUT,NBYTES )
       CALL GOPWK ( 1,TTOUT,WSTYPE )
       CALL GACWK ( 1 )
C
C      Center the window around the origin.
C
       CALL GSWN ( 1,-1.25,1.25,-1.25,1.25 )
       CALL GSELNT ( 1 )
C
C      Define the colors we'll be using
C
       CALL GSCR ( 1,0,0.0,0.0,1.0 )
       CALL GSCR ( 1,1,1.0,1.0,0.0 )
       CALL GSCR ( 1,2,1.0,1.0,1.0 )
C
C      Fill the star with solid yellow.
C
       CALL GSFAIS ( GSOLID )
       CALL GSFACI ( 1 )
C
C      Draw the star.
C
       CALL GFA ( 5,STARX,STARY )
C
C      Select large characters centered under the star.
C
       CALL GSCHH ( 0.15 )
       CALL GSTXAL ( GACENT,GAHALF )
       CALL GSTXCI ( 2 )
C
C      Draw the title.
C
       CALL GTX ( 0.0,-1.1,'Star' )
C
C      FORTRAN 77 subset version.
C      CALL GTXS ( 0.0,-1.1,4,'Star' )

C
C      Wait for break
C
       PRINT *, 'Done.  Enter BREAK in window...'
       CALL BWAIT(1)

C
C      Close the workstation and shutdown GKS.
C 
       CALL GDAWK ( 1 )
       CALL GCLWK ( 1 )
       CALL GCLKS
       STOP
       END


       SUBROUTINE BWAIT(IWK)
       CHARACTER*80	RECORD
       CALL GINCH(IWK, 1, 0, 1, 0.0, 1279.0, 0.0, 1023.0, 1, RECORD)
       CALL GSCHM(IWK, 1, 0, 1)
10     CALL GRQCH(IWK, 1, ISTAT, ICHNR)
       IF (ISTAT .NE. 0) GOTO 10
       END
