/*
 *		Copyright IBM Corporation 1989
 *
 *                      All Rights Reserved
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appear in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation, and that the name of IBM not be
 * used in advertising or publicity pertaining to distribution of the
 * software without specific, written prior permission.
 *
 * IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
 * ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
 * IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
 * ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
 * ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *
 * University of Illinois at Urbana-Champaign
 * Department of Computer Science
 * 1304 W. Springfield Ave.
 * Urbana, IL	61801
 *
 * (C) Copyright 1987, 1988 by The University of Illinois Board of Trustees.
 *	All rights reserved.
 *
 * Tool: X 11 Graphical Kernel System
 * Author: Gregory Scott Rogers
 * Author: Sung Hsien Ching Kelvin
 * Author: Yu Pan
 */

/* Converting to TMAP needs, making function within gksm2ps
 * J Davison 9.95
 *
 *
 *
 *
 *
 *
 */


#ifndef lint
    static char	rcsid[]	= "$Id$";
    static char	afsid[]	= "$__Header$";
#endif

/* Need to include wchar.h for RH9 - *kob* 10/03 */
#include <wchar.h>
#include "udposix.h"
#include <stdlib.h>
#include <stdio.h>
#include "xgks.h"
#include <gks_implem.h>
#include <math.h>

#define CLEAR_WORKSTATION	1

#define X_WSID			2	/* X workstation ID */
#define MO_WSID			3	/* Output Metafile workstation ID */

xpreview(int mi_wsid, int fileno, int nfile, char file[80])

{
    int		    ch,itemno;
    int		    error;
    int             c;
    int            first; /* First occurrence of WS viewport (item 72) */
    int            j, fktr;
    long	   maxsize	= 1024;
    char           *conn	= NULL;
    char           *mi_path;
    char           *mo_path	= NULL;
    int             answer;
    Gchar          *record	= (Gchar*) malloc(maxsize);
    Ggksmit         gksmit;
    extern char    *optarg;
    extern int      optind;
    Gcobundl        rep[7] = { {1.0,1.0,1.0},{0.0,0.0,0.0},{1.0,0.0,0.0},
				 {0.0,1.0,0.0},{0.0,0.0,1.0},
				 {0.0,1.0,1.0},{1.0,0.0,1.0} };
    Glnbundl        lnrep;
    XGKSMLIMIT     *limit;
    char           command[100];
    float          xsize_meta, ysize_meta, ref_len;
    float          aspect, xstretch, ystretch, xlen, ylen;

/*********************************************************************/

    if (record == NULL) {
	perror("malloc()");
	return 1;
    }

    /* Open set set up X workstation if first metafile */
    if(fileno == 0)
    {
      gopenws(X_WSID, conn, conn);
      gactivatews(X_WSID);
      printf("\n");
    }
    /* JD Set the background color -- WHITE FOR NOW ... */
    set_background(X_WSID,1);
    
    /* JD Set colors/styles for lines used */
    
    gsetcolorrep (X_WSID,0,&rep[0]);
    gsetcolorrep (X_WSID,1,&rep[1]);
    gsetcolorrep (X_WSID,2,&rep[2]);
    gsetcolorrep (X_WSID,3,&rep[3]);
    gsetcolorrep (X_WSID,4,&rep[4]);
    gsetcolorrep (X_WSID,5,&rep[5]);
    gsetcolorrep (X_WSID,6,&rep[6]);
    
    for (j = 1; j <= 6; j++){
      for (fktr = 1; fktr <= 3; fktr++){
	lnrep.type   = 1;
	lnrep.width  = (float) fktr; 
	lnrep.colour = j;
	
	gsetlinerep (X_WSID,(fktr-1)*6+j,&lnrep);
      }
    }
    
    lnrep.type   = 1;
    lnrep.width  = 1.0; 
    lnrep.colour = 0;
    gsetlinerep (X_WSID,19,&lnrep);

    /* Clear the workstation */
    gclearws(X_WSID, GALWAYS);

    for (itemno = 1, error = 0, first =1; !error; ++itemno) {
	int	status	= ggetgksm(mi_wsid, &gksmit);

	/* Handle last item in metafile */
	if (gksmit.type == 0)
	  break;

	if (status != 0) {
	    error	= status != 162;
	    break;
	}

	if (gksmit.type == CLEAR_WORKSTATION) {
	    Gwsdus         du;

	    (void) ginqwsdeferupdatest(X_WSID, &du);

	    if (du.dspsurf == GNOTEMPTY) {
		fprintf(stderr, "Exiting.  Please report your problem to the system manager.\n");
		exit(1);
	    }
	}

	if (gksmit.length > maxsize) {
	    maxsize	= gksmit.length;
	    if ((record = (Gchar*)realloc((voidp)record, maxsize)) == NULL) {
		perror("realloc()");
		error	= 1;
		break;
	    }
	}

	if ((status = greadgksm(mi_wsid, maxsize, record)) != 0) {
	    error	= status != 162;	/* EOF */
	    break;
	}

	/* JD Avoid interpretting type 34 -- Character vector setup */
	if (gksmit.type == 34)
	  continue;
	
	/* JD Ignore WS window and viewport calls after the first */
	if ((gksmit.type == 71 || gksmit.type == 72) && !first)
	  continue;

	/* Deal with inches in item 72 -- WS viewport setup */
	if (gksmit.type == 72)
	{
	  first = 0;
	  limit = (XGKSMLIMIT *) record;
	  xsize_meta = (*limit).rect.xmax;
	  ysize_meta = (*limit).rect.ymax;

	  aspect = ysize_meta / xsize_meta;
	  ystretch = (float) sqrt(aspect);
	  xstretch = 1.0 / ystretch;

	  /* Canonical Ferret plot size */
	  ref_len = (float) sqrt(10.2*8.8);

	  /* Now use 0.7 size factor */
	  xlen = xstretch*ref_len*0.8367;
	  ylen = ystretch*ref_len*0.8367; 

	  resize_xgks_window(X_WSID, xlen, ylen);
	  continue;
	}

	/* JD Ignore all segment handling */
	if (gksmit.type >= 81 && gksmit.type <= 84)
	  continue;

	error = ginterpret(&gksmit, record) != 0;
    }

    if (error)
	fprintf(stderr, "Error occured at item %d\n", itemno);

    (void) free((voidp)record);

    /* Delete this file? */
    printf(" -> Delete %s? (y/n) [n]: ", file);
    while ((ch = getchar()) != '\n')
      answer = tolower(ch);
    
    if (answer == 'y')
    {
      sprintf(command,"rm %s", file);
      system(command);
    }
   
    if (fileno + 1 == nfile)
    {
      printf("\n");
      gdeactivatews(X_WSID);
      gclosews(X_WSID);
    }

    return 0;
}

