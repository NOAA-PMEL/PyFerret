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
 *
 * University of Illinois at Urbana-Champaign
 * Department of Computer Science
 * 1304 W. Springfield Ave.
 * Urbana, IL	61801
 *
 * (C) Copyright 1987, 1988 by The University of Illinois Board of Trustees.
 * All rights reserved.
 *
 * Tool: X 11 Graphical Kernel System
 * Author: Gregory Scott Rogers
 * Author: Sung Hsien Ching Kelvin
 * Author: Yu Pan
 */

/* LINTLIBRARY */

#include "udposix.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "font.h"

#ifdef lint
    static void lint_malloc(n) size_t n; { n++; }
#   define	malloc(n)	(lint_malloc((n)), 0)
#else
    static char     rcsid[] = "$Id$";
    static char     afsid[] = "$__Header$";
#endif

#define MAXVC	6000

struct Font {
    char            fn[30];
    bits16          fnominalx, fnominaly;
    bits16          ftop, fcap, fhalf, fbase, fbottom;
    int             fc[256];
    struct vcharst  fvc[MAXVC];
} Font = {
    "FontName",
    0, 0,
    0, 0, 0, 0, 0,
    {
	-1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    },
    {
	{'e', 0, 0},
    }
};

static int             Vc = 0;		/* next available vcharst slot */
static struct vcharst *Vcp = Font.fvc;	/* pointer to same */
static FILE           *fontfile;
static int             xmin, ymin, xmax, ymax;
static int             cname;
static int             margin;
static int             default_margin;


    static void
SetCapHalf()
{
    struct vcharst *vcp;

    if (Font.fc['A'] != -1) {
	vcp = &(Font.fvc[Font.fc['A']]);
	while (vcp->vc_type != 'S')
	    vcp++;
	Font.fcap = vcp->vc_y;
    } else
	Font.fcap = Font.fnominaly;

    if (Font.fc['a'] != -1) {
	vcp = &(Font.fvc[Font.fc['a']]);
	while (vcp->vc_type != 'S')
	    vcp++;
	Font.fhalf = vcp->vc_y;
    } else
	Font.fhalf = Font.fnominaly / 2;

    Font.fbase = 0;
}


    static void
MakeSpaceChar()
{
    struct vcharst *vcpN;
    int             code;

    if (Font.fc['N'] != -1) {
	vcpN = &(Font.fvc[Font.fc['N']]);
	while (vcpN->vc_type != 'S')
	    vcpN++;
    } else
	vcpN = NULL;

    Font.fc[' '] = Vc;				/* Begin a SPACE charater */

    for (code = 1; code < 255; code++)		/* Make all invalid chars
						 * point to SPACE */
	if (Font.fc[code] == -1)
	    Font.fc[code] = Vc;

    Vcp->vc_type = 's';				/* extent minimum */
    Vcp->vc_x = 0;
    Vcp++->vc_y = 0;
    Vc++;

    Vcp->vc_type = 'S';				/* extent maximum */
    Vcp->vc_x = (vcpN != NULL) ? vcpN->vc_x : Font.fnominalx;
    Vcp++->vc_y = (vcpN != NULL) ? vcpN->vc_y : Font.fnominaly;
    Vc++;

    Vcp->vc_type = 'm';				/* commands to make a space */
    Vcp->vc_x = 0;
    Vcp++->vc_y = 0;
    Vc++;
    Vcp->vc_type = 'm';
    Vcp->vc_x = (vcpN != NULL) ? vcpN->vc_x : Font.fnominalx;
    Vcp++->vc_y = (vcpN != NULL) ? vcpN->vc_y : Font.fnominaly;
    Vc++;

    Vcp->vc_type = 'e';
    Vcp->vc_x = 0;
    Vcp++->vc_y = 0;
    Vc++;
}


    static void
WriteFont()
{
    int             fd;
    int             size;

    size = (char *) Vcp - (char *) &Font;

    if ((fd = open(Font.fn, O_CREAT | O_RDWR, 0644)) < 0) {
	perror("open");
	(void) exit(1);
    }
    if (write(fd, (char *) &size, sizeof(int)) != sizeof(int)) {
	perror("write size");
	(void) exit(1);
    }
    if (write(fd, (char *) &Font, size) != size) {
	perror("write Font");
	(void) exit(1);
    }
    (void) close(fd);
}


#if 0
    static void
ReadFont(Fpp, Fn)
    FONT          **Fpp;
    char           *Fn;
{
    int             fd;
    int             size;

    if ((fd = open(Fn, O_RDONLY, 0644)) < 0) {
	perror("open");
	(void) exit(1);
    }
    if (read(fd, (char *) &size, sizeof(int)) != sizeof(int)) {
	perror("read size");
	(void) exit(1);
    }
    if ((*Fpp = (FONT *) malloc((size_t) size)) == NULL) {
	perror("malloc failed");
	(void) exit(1);
    }
    if (read(fd, (char *) *Fpp, size) != size) {
	perror("read Font");
	(void) exit(1);
    }
    (void) close(fd);
}
#endif


#if 0
    static void
PrintFont(F)
    FONT           *F;
{
    int             c, co;
    struct vcharst *cp;

    (void) printf("Font name = %s\n", F->fname);
    (void) fflush(stdout);
    for (c = 0; c < 256; c++)
	if ((co = F->fcharoffset[c]) != -1) {
	    (void) printf("character %c [%d] : ", c, c);
	    for (cp = &(F->fchars[co]); (cp->vc_type != 'e'); cp++)
		(void) printf("{ '%c', %d, %d} ", cp->vc_type,
			      cp->vc_x, cp->vc_y);
	    (void) printf("\n");
	}
}
#endif


    static void
cklimits(x, y)
    int             x, y;
{
    if (x < xmin)
	xmin = x;
    if (x > xmax)
	xmax = x;
    if (y < ymin)
	ymin = y;
    if (y > ymax)
	ymax = y;

    if (y > Font.ftop)
	Font.ftop = y;
    if (y < Font.fbottom)
	Font.fbottom = y;
}


    static void
BeginChar(s)
    char           *s;
{
    margin	= default_margin;

    xmin = ymin = 5000;
    xmax = ymax = -5000;

    if (strlen(s) == 1)
	cname = *s;
    else
	cname = 128 + *(++s);

    Font.fc[cname] = Vc;
    Vcp++;
    Vc++;					/* save room for min */
    Vcp++;
    Vc++;					/* save room for max */
}


    static void
FinishChar()
{
    struct vcharst *vp;

    vp = &(Font.fvc[Font.fc[cname]]);
    vp->vc_type = 's';
    vp->vc_x = xmin - margin;
    vp++->vc_y = ymin - margin;

    vp->vc_type = 'S';
    vp->vc_x = xmax + margin;
    vp->vc_y = ymax + margin;

    Vcp->vc_type = 'e';
    Vcp->vc_x = 0;
    Vcp++->vc_y = 0;
    Vc++;

    if ((xmax - xmin) + 2*margin > Font.fnominalx)
	Font.fnominalx = (xmax - xmin) + 2*margin;
    if ((ymax - ymin) + 2*margin > Font.fnominaly)
	Font.fnominaly = (ymax - ymin) + 2*margin;
}


    static void
ReadVFont(argc, argv)
    int             argc;
    char           *argv[];
{
    int             x, y, spacing, width, i;
    char            s[80], *sp;

    if (argc < 3) {
	(void) fprintf(stderr, "usage:mkfont vfontin gksfontout\n");
	(void) exit(1);
    }
    if ((fontfile = fopen(argv[1], "r")) == NULL) {
	(void) fprintf(stderr, "can't fopen(%s,\"r\")\n", argv[1]);
	(void) exit(1);
    }
    (void) strncpy(Font.fn, argv[2], 30);	/* font name */
    /*
     * Make sure every char is undefined.
     */
    for (i = 0; i < 256; i++)
	Font.fc[i] = -1;

    Font.ftop = -9999;
    Font.fbottom = 9999;
    Font.fnominalx = 0;
    Font.fnominaly = 0;

    while (fgets(s, 80, fontfile) != NULL) {
	sp = s;
	switch (*sp++) {
	case 'S':
	    /* (void)fprintf(stdout,"Special font\n"); */
	    break;
	case 'U':
	    (void) sscanf(sp, "%d", &spacing);
	    default_margin	= spacing/2;
	    break;
	case 'u':
	    (void) sscanf(sp, "%d", &spacing);
	    margin	= spacing/2;
	    break;
	case 'C':
	    sp++;
	    sp[strlen(sp) - 1] = NULL;		/* past the space & null */
	    /* (void)fprintf(stdout,"Character name = %s\n",sp); */
	    BeginChar(sp);
	    break;
	case 'w':
	    (void) sscanf(sp, "%d", &width);
	    /* (void)fprintf(stdout,"total width = %d\n",width); */
	    break;
	case 'm':
	    (void) sscanf(sp, "%d%d", &x, &y);
	    x += margin;
	    y += margin;
	    cklimits(x, y);
	    Vcp->vc_type = 'm';
	    Vcp->vc_x = x;
	    Vcp++->vc_y = y;
	    Vc++;

	    break;
	case 'n':
	    (void) sscanf(sp, "%d%d", &x, &y);
	    x += margin;
	    y += margin;
	    cklimits(x, y);
	    Vcp->vc_type = 'd';
	    Vcp->vc_x = x;
	    Vcp++->vc_y = y;
	    Vc++;

	    break;
	case 'E':
	    /* (void)fprintf(stdout,"End of that one...."); */
	    FinishChar();
	    break;
	default:
	    /* (void)fprintf(stdout,"%s\n",s);	 */
	    break;
	}
    }
}


main(argc, argv)
    int             argc;
    char           *argv[];
{
    ReadVFont(argc, argv);

    SetCapHalf();

    MakeSpaceChar();

    WriteFont();

    return 0;
}
