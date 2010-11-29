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

#ifndef lint
    static char	rcsid[]	= "$Id$";
    static char	afsid[]	= "$__Header$";
#endif

#include "udposix.h"
#include <stdlib.h>
#include <stdio.h>
#include "xgks.h"
#include "demo.h"

#define CLEAR_WORKSTATION	1

#define MI_WSID			1	/* Input Metafile workstation ID */
#define X_WSID			2	/* X workstation ID */
#define MO_WSID			3	/* Output Metafile workstation ID */


main(argc, argv)
    int             argc;
    char           *argv[];
{
    int		    itemno;
    int		    error;
    int             c;
    long	   maxsize	= 1024;
    char           *conn	= NULL;
    char           *mi_path;
    char           *mo_path	= NULL;
    Gchar          *record	= (Gchar*) malloc(maxsize);
    Ggksmit         gksmit;
    extern char    *optarg;
    extern int      optind;

    if (record == NULL) {
	perror("malloc()");
	return 1;
    }

    while ((c = getopt(argc, argv, "d:o:")) != -1) {
	switch (c) {
	case 'd':
	    conn	= optarg;
	    break;
	case 'o':
	    mo_path	= optarg;
	}
    }

    if (optind >= argc) {
	fprintf(stderr, "usage: %s [-d display] [-o mo_path] file\n", argv[0]);
	exit(0);
    }

    mi_path	= argv[optind];

    gopengks(stderr, 0);

    gopenws(MI_WSID, mi_path, "MI");
    gopenws(X_WSID, conn, conn);
    if (mo_path != NULL)
	gopenws(MO_WSID, mo_path, "MO");

    gactivatews(X_WSID);
    if (mo_path != NULL)
	gactivatews(MO_WSID);

    for (itemno = 1, error = 0; !error; ++itemno) {
	int	status	= ggetgksm(MI_WSID, &gksmit);

	if (status != 0) {
	    error	= status != 162;
	    break;
	}

	if (gksmit.type == CLEAR_WORKSTATION) {
	    Gwsdus         du;

	    (void) ginqwsdeferupdatest(X_WSID, &du);

	    if (du.dspsurf == GNOTEMPTY) {
		fprintf(stderr, "Hit BREAK in window to continue\n");
		WaitForBreak(X_WSID);
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

	if ((status = greadgksm(MI_WSID, maxsize, record)) != 0) {
	    error	= status != 162;	/* EOF */
	    break;
	}
	error	= ginterpret(&gksmit, record) != 0;
    }
    if (error)
	fprintf(stderr, "Error occured at item %d\n", itemno);

    (void) free((voidp)record);

    fprintf(stderr, "Done.  Hit BREAK in window to quit.\n");

    WaitForBreak(X_WSID);

    gdeactivatews(X_WSID);
    if (mo_path != NULL)
	gdeactivatews(MO_WSID);

    gclosews(X_WSID);
    gclosews(MI_WSID);
    if (mo_path != NULL)
	gclosews(MO_WSID);

    gclosegks();

    return 0;
}
