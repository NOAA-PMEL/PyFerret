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
 *
 * Mod to GetWindowGeometry to change default window size *jd* 3.21.94
 * Mod to AllocGroundColors to set background/foreground defaults *jd* 3.21.94
 * Added WindowMapping to control output window visibility *sh* 16-sep-94
 *
 * XGKS doesn't work correctly on non-indexed color displays. So, the quick
 * fix utilized here is to 1) see if the default visual uses a color table;
 * if so, use that. 2) If not, we look for the best color table (if any) to
 * use. If there are no indexed visuals, we're out of luck *js* 7.29.97
 *
 * Disabled quick fix listed above due to fix of XGKS code to support 
 * non-indexed color visuals
 *
 * XGKS open workstation function
 * Open display wk->conn if necessary, then create and map the workstation 
 * window wk: workstation list pointer return: (0)---- open succeeds (21)-- 
 * open fails .
 *
 * 2.17.98 Modified workstation initialization code to use XgksMaxColours
 * to avoid setting the number of colors used in two places *js* 
 * 
 */

/*LINTLIBRARY*/

#include "udposix.h"
#include <assert.h>
#include <stdlib.h>
#include <sys/types.h>			/* for uid_t */
#include <unistd.h>			/* NeXTOS requires that this be */
					/* after <sys/types.h> */
#include <string.h>
#include <pwd.h>
#include <ctype.h>			/* for isupper() & tolower() */
#include "gks_implem.h"
#include <X11/Xresource.h>

#ifdef lint
    static void	lint_malloc(n) size_t n; { n++; }
#   define	malloc(n)	(lint_malloc((n)), 0)
#else
    static char afsid[]	= "$__Header$";
    static char rcsid[]	= "$Id$";
#endif

static int      InitDefaults();
static int      GetUsersDatabase();
static int      InsureConn();
static int      CreateWindow();
static char    *GetHomeDir();
static void     GetWindowGeometry();
static void     AllocGroundColors();
static void     GetWMHints();
static void     CreateGC();
static void     UpdateOpenWSTable();

/* ******** *sh* 16-sep-94 code added for external control of window mapping */
/* further mods inside xXgksOpenWs */

static int map_that_window=1;  /* static external */

/*
use WindowMapping(0) for invisible windows; WindowMapping(1) for mapped ones.
*/

void WindowMapping( map_it )
int map_it;
{
  map_that_window = map_it;
  return;
}

/* ******** end of 16-sep-94 code */




/*
 * The following may be changed by applications programs to the name of
 * the application.
 */
extern char    *progname;


    Gint
xXgksOpenWs(wk)
    WS_STATE_PTR    wk;
{
    int             status;		/* success */

    if (wk->ewstype != X_WIN) {
	status = INVALID;
    } else {

	/* Insure connection to display. */
	if (!(status = InsureConn(wk))) {
	    char           *basename;	/* UPC: base program name */
	    XrmDatabase     rDB;	/* resource database */

	    (void) xXgksSIGIOStart(wk);
	    (void) XgksSIGIO_OFF(wk->dpy);

	    /* Extract program name. */
	    if (progname == NULL)
		progname = "XGKS";
	    if ((basename = strrchr(progname, '/')) == (char *) NULL)
		basename = progname;
	    else
		basename += 1;

	    /* Initialize resource database (i.e. X-defaults). */
	    (void) InitDefaults(basename, wk->dpy, &rDB);

	    /* Create window -- using resource database for defaults. */
	    if (!(status = CreateWindow(basename, rDB, wk))) {
		XEvent          xev;	/* X event structure */
		XWindowAttributes WinAtt;	/* window attributes */

		/* Map window.  Wait for exposure-event. */

/* *sh* 16-sep-94: bypass this code for invisible, unmapped windows */
		if ( map_that_window ) {
		  XMapWindow(wk->dpy, wk->win);
		  XWindowEvent(wk->dpy, wk->win, ExposureMask, &xev);
		  XSync(wk->dpy, 0);
		}

		/* Get size of actual window obtained. */
		XGetWindowAttributes(wk->dpy, wk->win, &WinAtt);
		wk->wbound.x = WinAtt.width;
		wk->wbound.y = WinAtt.height;

		/* Update open-workstation table. */
		(void) UpdateOpenWSTable(wk);

		/* Select Input Events */
		XSelectInput(wk->dpy, wk->win, wk->event_mask);
		XSync(wk->dpy, 0);
	    }
	}
	/*
	 * We depend on XgksSIGIO_ON() checking for a NULL wk->dpy -- caused
	 * by failure to establish a connection.
	 */
	(void) XgksSIGIO_ON(wk->dpy);
    }

    return status ? status : OK;
}


/*
 * Insure a connection to the display.
 */
    static
InsureConn(wk)
    WS_STATE_PTR    wk;
{
    int             status = 0;		/* success */
    int             i;
    WS_STATE_PTR    wk_p = NULL;

    /* Check if Display wk->conn has been opened  */
    for (i = 0; i < MAX_OPEN_WS; i++) {
	if (xgks_state.openedws[i].ws_id >= 0 &&
		xgks_state.openedws[i].ws->ewstype == X_WIN) {

	    WS_STATE_PTR    ws = xgks_state.openedws[i].ws;

	    if ((STRCMP(wk->conn, ws->conn) == 0) && (ws != wk)) {
		wk_p = ws;
		break;
	    }
	}
    }

    if (wk_p != NULL) {				/* Has been opened */
	wk->dpy = wk_p->dpy;
	wk->dclmp = wk_p->dclmp;
	wk->wclmp = wk_p->dclmp;
    } else {					/* Open a new display */
	if ((wk->dpy = XOpenDisplay(wk->conn)) == NULL) {
	    status = 26;
	} else {
	    char           *ptr = DisplayString(wk->dpy);

	    ufree((voidp)wk->conn);
	    if ((wk->conn = (char *) malloc((size_t) (STRLEN(ptr) + 1))) == 
		    NULL) {
		status = 300;
	    } else {
		STRCPY(wk->conn, ptr);

		XSelectInput(wk->dpy, DefaultRootWindow(wk->dpy),
			     0);
	    }
	}
    }

    return status;
}


/*
 *	Indicate whether a boolean resource is set.  Returns -1 if the
 *	resource doesn't exist, 0 if it's false, and 1 if it's true.
 */
    static int
BoolResource(prog, name, class, rDB)
    char           *prog;		/* program name */
    char           *name;		/* resource name */
    char           *class;		/* resource class */
    XrmDatabase     rDB;		/* resource database */
{
    int             ison = -1;		/* return status = doesn't exist */

    char            name_key[1024];
    char            class_key[1024];
    char           *str_type[20];
    XrmValue        value;

    (void) strcat(strcat(strcpy(name_key, prog), "."), name);
    (void) strcat(strcpy(class_key, "Xgks."), class);

    if (XrmGetResource(rDB, name_key, class_key, str_type, &value) == True) {
	char           *cp;

	(void) strncpy(name_key, value.addr, (int) value.size);
	name_key[value.size] = 0;

	for (cp = name_key; *cp != 0; ++cp) {
	    if (isupper(*cp))
		*cp = tolower(*cp);
	}

	ison =
	    strcmp(name_key, "on") == 0 ||
	    strcmp(name_key, "1") == 0 ||
	    strcmp(name_key, "yes") == 0 ||
	    strcmp(name_key, "set") == 0 ||
	    strcmp(name_key, "true") == 0;
    }
    return ison;
}

XVisualInfo *getBestVisual(Display *dpy, int *index)
{
  XVisualInfo *visualList;
  XVisualInfo     visualTemplate;
  int numMatched;

  /* Try default visual first */
  visualTemplate.screen = DefaultScreen(dpy);
  visualTemplate.visualid =
    XVisualIDFromVisual(DefaultVisual(dpy, DefaultScreen(dpy)));
  visualList =
    XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask, &visualTemplate,
		   &numMatched);
  *index = 0;
  return visualList;
}


/* getBestVisual has been deprecated, due to fix of direct mapped color
   problems
*/
#if 0
static int VisualsToUse[] = {
  PseudoColor, StaticColor, GrayScale, StaticGray
};

static XVisualInfo *getBestVisual(Display *dpy, int *index)
{
  XVisualInfo *visualList;
  XVisualInfo     visualTemplate;
  int numMatched;

  /* Try default visual first */
  visualTemplate.screen = DefaultScreen(dpy);
  visualTemplate.visualid =
    XVisualIDFromVisual(DefaultVisual(dpy, DefaultScreen(dpy)));
  visualList =
    XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask, &visualTemplate,
		   &numMatched);
  *index = 0;

  if (visualList == NULL || visualList->class == TrueColor ||
      visualList->class == DirectColor){	/* Plan B -- get best supported visual */
    int class, i;
    visualList = NULL;
    for (i=0; i < sizeof(VisualsToUse)/sizeof(int); ++i){
      visualTemplate.class = VisualsToUse[i];
      visualList =
	XGetVisualInfo(dpy,VisualScreenMask | VisualClassMask,
		       &visualTemplate, &numMatched);
      if (visualList != NULL)
	break;
    }
    if (visualList){
      /* Get deepest visual */
      int j;
      int maxDepth = -1;
      for (j=0; j < numMatched; ++j){
	if (visualList[j].depth > maxDepth){
	  maxDepth = visualList[j].depth;
	}
      }
      assert(maxDepth > 0);
      *index = i;
    }
  }
  return visualList;
}
#endif

/*
 * Create an XGKS window -- with all its associated attributes.
 */
    static
CreateWindow(name, rDB, wk)
    char           *name;		/* program name */
    XrmDatabase     rDB;		/* resource database */
    WS_STATE_PTR    wk;			/* workstation structure */
{
    int             status = 0;		/* success */
    int             NumMatched;		/* number of visuals found */
    Display        *dpy = wk->dpy;	/* for convenience */
    XSizeHints      SizeHints;		/* window size hints */
    XSetWindowAttributes xswa;		/* window attributes */
    XVisualInfo    *VisualList, *theVisual;
    int visualIndex;


    VisualList = getBestVisual(dpy, &visualIndex);
    
    /* size of colour map */
    if (VisualList == NULL || !XcInit(wk, &VisualList[visualIndex])) {
      fprintf(stderr, "Ferret cannot run on this X server\n");
      exit(1);
    } else {
      theVisual = &VisualList[visualIndex];
      /* Set the screen default colour map ID. */
      if (theVisual->visual == DefaultVisual(wk->dpy, DefaultScreen(wk->dpy))){
	wk->dclmp = DefaultColormap(wk->dpy,
				    DefaultScreen(wk->dpy));
      } else {
	wk->dclmp = XCreateColormap(wk->dpy, DefaultRootWindow(wk->dpy),
				    theVisual->visual, AllocNone);
      }
    /* Initialize color-mapping. */
      wk->wscolour = 0;
      wk->wscolour = XgksMaxColours(wk->wstype);
      wk->wclmp = wk->dclmp;

	/*
	 * Get foreground and background colors (and set them in the
	 * colormap).
	 */
	(void) AllocGroundColors(wk, name, rDB, &wk->wsfg, &wk->wsbg);

	/*
	 * Init pointer to table of set values to NULL.  The table will be
	 * alloc'ed and init'ed in gsetcolourrep on the 1st call (DWO).
	 * 
	 * This may no longer be meaningful since I've changed the way the
	 * mapping from GKS color-index to X-colorcell-index is handled.
	 * --SRE 2/1/90
	 */
	wk->set_colour_rep = (Gcobundl *) NULL;

	/* Get window geometry hints. */
	(void) GetWindowGeometry(name, dpy, rDB, &SizeHints);

	/* Set the window-event mask. */
	wk->event_mask = StructureNotifyMask | 
	    ExposureMask | 
	    KeyPressMask |
	    ButtonPressMask | ButtonReleaseMask | ButtonMotionMask;

	/*
	 * Set the window-attributes structure.
	 */
	xswa.event_mask = StructureNotifyMask | ExposureMask;
	xswa.background_pixel = XcPixelValue(wk, wk->wsbg);
	xswa.border_pixel = XcPixelValue(wk, wk->wsfg);
	xswa.do_not_propagate_mask = wk->event_mask &
	    (KeyPressMask | KeyReleaseMask | ButtonPressMask |
	     ButtonReleaseMask | PointerMotionMask | Button1MotionMask |
	     Button2MotionMask | Button3MotionMask | Button4MotionMask |
	     Button5MotionMask | ButtonMotionMask);
	xswa.colormap = wk->dclmp;		/* default colormap --SRE */
	xswa.backing_store	= 
	    BoolResource(name, "backingstore", "Backingstore", rDB) == 0
		? NotUseful
		: DoesBackingStore(DefaultScreenOfDisplay(dpy)) == NotUseful
		    ? NotUseful
		    : Always;

	/* Create the window. */
	if ((wk->win = XCreateWindow(dpy, DefaultRootWindow(dpy),
				     (int) SizeHints.x, (int) SizeHints.y,
			      (int) SizeHints.width, (int) SizeHints.height,
				     5,		/* border width */
				     VisualList[visualIndex].depth,
			InputOutput, VisualList[visualIndex].visual,
	    (unsigned long) (CWDontPropagate | CWBackPixel | CWBorderPixel |
	     CWEventMask | CWColormap | CWBackingStore), &xswa)) == False) {

	    status = 26;
	} else {
	    Window          win = wk->win;	/* for convenience */
	    XWMHints        WMHints;	/* window-manager hints */
	    XClassHint      ClassHints;	/* class hints */

	    /* Set standard window properties. */
	    XSetStandardProperties(dpy, win, name, name, None,
				   (char **) NULL, 0, &SizeHints);

	    /* Set window-manager hints. */
	    (void) GetWMHints(dpy, name, rDB, &WMHints);
	    XSetWMHints(dpy, win, &WMHints);

	    /* Set class hints. */
	    if ((ClassHints.res_name = getenv("RESOURCE_NAME")) == NULL)
		ClassHints.res_name = name;
	    ClassHints.res_class = name;
	    (void) XSetClassHint(dpy, win, &ClassHints);

	    /* Create graphics-context for window. */
	    (void) CreateGC(dpy, win, wk);

	    /* Set foreground and background colors in graphics-context. */
	    XSetForeground(dpy, wk->gc, XcPixelValue(wk, wk->wsfg));
	    XSetBackground(dpy, wk->gc, XcPixelValue(wk, wk->wsbg));

	    /*
	     * Initialize last-clipping rectangles to absurd values.  This
	     * will cause actual clipping window to be set.
	     */
	    wk->last_pline_rectangle.x = 0;
	    wk->last_pline_rectangle.y = 0;
	    wk->last_pline_rectangle.width = 0;
	    wk->last_pline_rectangle.height = 0;
	    wk->last_pmarker_rectangle = wk->last_pline_rectangle;
	    wk->last_farea_rectangle = wk->last_pline_rectangle;
	    wk->last_text_rectangle = wk->last_pline_rectangle;

	    wk->last_dash_index = 1;

	    /* Set soft-clipping if appropriate.  It's off by default. */
	    wk->soft_clipping_on = BoolResource(name, "softclipping",
						"Softclipping", rDB) == 1;

	    /* Save the setting of backing-store. */
	    wk->backing_store_on	= xswa.backing_store == Always;
	}					/* window created */
    }						/* color-mapping initialized */

    if (VisualList != NULL){
      XFree((char *)VisualList);
    }
	
    return status;
}


/*
 * Create a graphics-context for a window.
 */
    static void
CreateGC(dpy, win, wk)
    Display        *dpy;
    Window          win;
    WS_STATE_PTR    wk;
{
    wk->gc = XCreateGC(dpy, win, (unsigned long) 0,
		       (XGCValues *) NULL);
    wk->plinegc = XCreateGC(dpy, win, (unsigned long) 0,
			    (XGCValues *) NULL);
    wk->pmarkgc = XCreateGC(dpy, win, (unsigned long) 0,
			    (XGCValues *) NULL);
    wk->fillareagc = XCreateGC(dpy, win, (unsigned long) 0,
			       (XGCValues *) NULL);
    wk->textgc = XCreateGC(dpy, win, (unsigned long) 0,
			   (XGCValues *) NULL);
}


/*
 * Update the open-workstation table by saving the window-identifier.
 *
 * I don't like the fact that this routine assumes that the appropriate
 * slot exists -- SRE.
 */
    static void
UpdateOpenWSTable(wk)
    WS_STATE_PTR    wk;
{
    int             i;

    for (i = 0; i < MAX_OPEN_WS; i++)
	if (wk->ws_id == xgks_state.openedws[i].ws_id)
	    break;
    xgks_state.openedws[i].win = wk->win;
}


/*
 * Get window geometry defaults.
 */
    static void
GetWindowGeometry(name, dpy, rDB, SizeHints)
    char           *name;		/* program name */
    Display        *dpy;
    XrmDatabase     rDB;		/* resource database */
    XSizeHints     *SizeHints;		/* window-size hints */
{
    char            buf[1024];
    char           *str_type[20];
    XrmValue        value;
    float           xf, yf;

    SizeHints->flags = 0;

    /*
     * Set to program-specified values and then override with any
     * user-specified values.
     */

    xf = ((float) DisplayWidth(dpy, DefaultScreen(dpy))) / ((float) DisplayWidthMM(dpy, DefaultScreen(dpy)));

    yf = ((float) DisplayHeight(dpy, DefaultScreen(dpy)))/((float) DisplayHeightMM(dpy, DefaultScreen(dpy)));

    /* Size window to (10.2,8.8)*sqrt(0.7) inches */

    SizeHints->width  = xf * 8.533932 * 25.4; /* was 640 */
    SizeHints->height = yf * 7.362608 * 25.4; /* was 512 */

    SizeHints->x = (DisplayWidth(dpy, DefaultScreen(dpy)) -
		    SizeHints->width) >> 1;
    SizeHints->y = (DisplayHeight(dpy, DefaultScreen(dpy)) -
		    SizeHints->height) >> 1;
    SizeHints->flags |= PSize | PPosition;

    if (XrmGetResource(rDB, strcat(strcpy(buf, name), ".geometry"),
		       "Xgks.Geometry", str_type, &value) == True) {

	int             x, y;
	long            flags;
	unsigned        width, height;

	(void) strncpy(buf, value.addr, (int) value.size);

	flags = XParseGeometry(buf, &x, &y, &width, &height);

	if (WidthValue & flags && HeightValue & flags &&
	     width >= 1 && width <= DisplayWidth(dpy, DefaultScreen(dpy)) &&
	  height >= 1 && height <= DisplayHeight(dpy, DefaultScreen(dpy))) {

	    SizeHints->width = width;
	    SizeHints->height = height;
	    SizeHints->flags |= USSize;
	}
	if (XValue & flags && YValue & flags) {
	    if (XNegative & flags)
		x += DisplayWidth(dpy, DefaultScreen(dpy)) -
		    SizeHints->width;
	    if (YNegative & flags)
		y += DisplayHeight(dpy, DefaultScreen(dpy)) -
		    SizeHints->height;
	    SizeHints->x = x;
	    SizeHints->y = y;
	    SizeHints->flags |= USPosition;
	}
    }
}


/*
 * Get foreground and background color defaults.
 */
    static void
AllocGroundColors(wk, name, rDB, fg, bg)
    WS_STATE_PTR    wk;			/* workstation structure */
    char           *name;		/* program name */
    XrmDatabase     rDB;		/* resource database */
    Gint           *fg, *bg;		/* fore/back-ground indexes */
{
    char            buf[1024];
    char           *str_type[20];
    XrmValue        value;

    if (BoolResource(name, "invertmono", "Invertmono", rDB) == 1 ||
	    BoolResource(name, "reverse", "Reverse", rDB) == 1) {

	*fg = 0;
	*bg = 1;
    } else {
	*fg = 1;
	*bg = 0;
    }

    /*
     * Set XGKS background color.
     */

    if (1) /* Set background to white by default */ {

	    Gcobundl        GKSrep;

	    GKSrep.red   = 1.0;
	    GKSrep.green = 1.0;
	    GKSrep.blue  = 1.0;

	    (void) XcSetColour(wk, (Gint) 0, &GKSrep);
	}
/*
    if (XrmGetResource(rDB, strcat(strcpy(buf, name), ".background"),
		       "Xgks.Background", str_type, &value) == True) {

	XColor          Xrep;

	(void) strncpy(buf, value.addr, (int) value.size);

	if (XParseColor(wk->dpy, DefaultColormap(wk->dpy,
			DefaultScreen(wk->dpy)), buf, &Xrep)) {

	    Gcobundl        GKSrep;

	    GKSrep.red   = (double)Xrep.red   / 65535.0;
	    GKSrep.green = (double)Xrep.green / 65535.0;
	    GKSrep.blue  = (double)Xrep.blue  / 65535.0;

	    (void) XcSetColour(wk, (Gint) 0, &GKSrep);
	} else {
	    (void) fprintf(stderr, "%s\"%s\"%s\n",
			   "AllocGroundColors: Background color ", buf,
			   " not known.  Using default.");
	}
    }
*/

    /*
     * Set XGKS foreground color.
     */


    if (1) /* Set foreground to black by default */ {

	    Gcobundl        GKSrep;

	    GKSrep.red   = 0.0;
	    GKSrep.green = 0.0;
	    GKSrep.blue  = 0.0;

	    (void) XcSetColour(wk, (Gint) 1, &GKSrep);
	}

/*
    if (XrmGetResource(rDB, strcat(strcpy(buf, name), ".foreground"),
		       "Xgks.Foreground", str_type, &value) == True) {

	XColor          Xrep;

	(void) strncpy(buf, value.addr, (int) value.size);

	if (XParseColor(wk->dpy, DefaultColormap(wk->dpy,
				     DefaultScreen(wk->dpy)), buf, &Xrep)) {

	    Gcobundl        GKSrep;

	    GKSrep.red   = (double)Xrep.red   / 65535.0;
	    GKSrep.green = (double)Xrep.green / 65535.0;
	    GKSrep.blue  = (double)Xrep.blue  / 65535.0;

	    (void) XcSetColour(wk, (Gint) 1, &GKSrep);
	} else {
	    (void) fprintf(stderr, "%s\"%s\"%s\n",
			   "AllocGroundColors: Foreground color ", buf,
			   " not known.  Using default.");
	}
    }
*/
}


/*
 * Get window-manager hints.
 */
    static void
GetWMHints(dpy, name, rDB, WMHints)
    Display        *dpy;		/* display */
    char           *name;		/* program name */
    XrmDatabase     rDB;		/* resource database */
    XWMHints       *WMHints;		/* window-manager hints */
{
    char            buf[1024];
    char           *str_type[20];
    XrmValue        value;

    WMHints->flags = 0;

    WMHints->input = True;
    WMHints->flags |= InputHint;

    WMHints->initial_state =
	BoolResource(name, "iconic", "Iconic", rDB) == 1 ? IconicState
	: NormalState;
    WMHints->flags |= StateHint;

    if (XrmGetResource(rDB, strcat(strcpy(buf, name), ".icon.geometry"),
		       "Xgks.Icon.Geometry", str_type, &value) == True) {

	int             x, y;
	long            flags;
	unsigned        width, height;

	(void) strncpy(buf, value.addr, (int) value.size);

	flags = XParseGeometry(buf, &x, &y, &width, &height);

	if (XValue & flags && YValue & flags) {
	    if (XNegative & flags && !((WidthValue & flags)) ||
		    YNegative & flags && !((HeightValue & flags))) {

		(void) fprintf(stderr, "%s%s\n",
			       "GetWMHints: Negative X (Y) icon position ",
			       "requires height (width) spec.");
	    } else {
		if (XNegative & flags)
		    x += DisplayWidth(dpy, DefaultScreen(dpy)) - width;
		if (YNegative & flags)
		    y += DisplayHeight(dpy, DefaultScreen(dpy)) - height;

		WMHints->icon_x = x;
		WMHints->icon_y = y;
		WMHints->flags |= IconPositionHint;
	    }
	}
    }
}


/*
 * Initialize our local resource manager.  Taken from "X11R4/contrib/
 * examples/OReilly/Xlib/basecalc/basecalc.c".
 */
    static
InitDefaults(name, dpy, rDB)
    char           *name;		/* name of application */
    Display        *dpy;
    XrmDatabase    *rDB;		/* resource database */
{
    int             status = 1;		/* routine status = success */

    /* So we can use the resource manager data-merging functions */
    XrmInitialize();

    /* Clear resource database */
    *rDB = XrmGetStringDatabase("");

    /* Get server defaults, program defaults, and .Xdefaults and merge them */
    (void) GetUsersDatabase(name, rDB, dpy);

    return status;
}


/*
 * Get program's and user's defaults
 */
    static
GetUsersDatabase(prog, rDB, dpy)
    char           *prog;
    XrmDatabase    *rDB;		/* resource database */
    Display        *dpy;
{
    int             status = 1;		/* routine status = success */
    XrmDatabase     homeDB, serverDB, applicationDB;

    char            filename[1024];
    char           *environment;
    char            name[255];
    char           *appresdir = getenv("XAPPLRESDIR");

    if (appresdir == NULL)
	appresdir = "/usr/lib/X11/app-defaults";

    (void) strcpy(name, appresdir);
    (void) strcat(name, "/");
    (void) strcat(name, prog);

    /*
     * Get application defaults file, if any.
     */
    applicationDB = XrmGetFileDatabase(name);
    (void) XrmMergeDatabases(applicationDB, rDB);

    /*
     * MERGE server defaults, these are canonically created by xrdb, loaded 
     * as a property of the root window when the server initializes.  If not 
     * defined, then use the resources specified in ~/.Xdefaults.
     */
    {
	int		actual_format;
	Atom		actual_type;
	unsigned long	nitems;
	unsigned long	bytesafter;
	unsigned char	*prop;

	if (XGetWindowProperty(dpy, DefaultRootWindow(dpy), 
			       XA_RESOURCE_MANAGER, 
			       (long)0, (long)0, False,
			       XA_STRING, &actual_type, &actual_format, 
			       &nitems, &bytesafter, &prop) == Success
		&& actual_type == XA_STRING) {
	    (void) XGetWindowProperty(dpy, DefaultRootWindow(dpy), 
				      XA_RESOURCE_MANAGER, 
				      (long)0, (long)bytesafter, False,
				      XA_STRING, &actual_type, &actual_format, 
				      &nitems, &bytesafter, &prop);
	    serverDB = XrmGetStringDatabase((char*)prop);

	} else {
	    /*
	     * Read ~/.Xdefaults file.
	     */
	    (void) GetHomeDir(filename);
	    (void) strcat(filename, "/.Xdefaults");
	    serverDB = XrmGetFileDatabase(filename);
	}
    }
    XrmMergeDatabases(serverDB, rDB);

    /*
     * Open XENVIRONMENT file, or if not defined, the
     * ~/.Xdefaults-<hostname>, and merge into existing data base
     */
    if ((environment = getenv("XENVIRONMENT")) == NULL) {
	int             len;

	environment = GetHomeDir(filename);
	(void) strcat(environment, "/.Xdefaults-");
	len = strlen(environment);
	(void) gethostname(environment + len, sizeof(filename) - len);
    }
    homeDB = XrmGetFileDatabase(environment);
    XrmMergeDatabases(homeDB, rDB);

    return status;
}


/*
 * Get the path of the user's home directory.
 */
    static char*
GetHomeDir(dest)
    char           *dest;
{
    uid_t           uid;
    struct passwd  *pw;
    register char  *ptr;

    if ((ptr = getenv("HOME")) != NULL) {
	(void) strcpy(dest, ptr);

    } else {
	if ((ptr = getenv("USER")) != NULL) {
	    pw = getpwnam(ptr);
	} else {
	    uid = getuid();
	    pw = getpwuid(uid);
	}
	if (pw) {
	    (void) strcpy(dest, pw->pw_dir);
	} else {
	    *dest = '\0';
	}
    }
    return dest;
}


/*
 *  xXgksClearWs(wk) --- clear the corresponding x-window
 */
xXgksClearWs(wk)
    WS_STATE_PTR    wk;
{
    if (wk->ewstype != X_WIN)
	return OK;

    (void) XgksSIGIO_OFF(wk->dpy);
    XClearArea(wk->dpy, wk->win, 0, 0, 0, 0, False);
    XSync(wk->dpy, 0);
    (void) XgksSIGIO_ON(wk->dpy);

    return OK;
}


/*
 * xXgksCloseWs(ws) --- close the corresponding x-window
 */
xXgksCloseWs(ws)
    WS_STATE_PTR    ws;
{
    if (ws->ewstype != X_WIN)
	return OK;

    (void) XgksSIGIO_OFF(ws->dpy);
    XUnmapWindow(ws->dpy, ws->win);
    XDestroyWindow(ws->dpy, ws->win);
    XFreeGC(ws->dpy, ws->gc);
    XSync(ws->dpy, 0);
    (void) XcEnd(ws);				/* free color-index
						 * mapping-thingy */
    (void) XgksSIGIO_ON(ws->dpy);

    return OK;
}


/*
 * xXgksHighLight(ws, bd) --- heighlight a primitive
 */
xXgksHighLight(ws, bd)
    Gpoint         *bd;
    WS_STATE_PTR    ws;
{
    Display        *dpy;
    Window          win;
    GC              gc;

    XPoint          box[5];
    int             i;

    dpy = ws->dpy;
    win = ws->win;
    gc = ws->gc;

    if (ws->ewstype != X_WIN)
	return OK;

    (void) XgksSIGIO_OFF(ws->dpy);
    XSetFunction(dpy, gc, GXinvert);
    XSetLineAttributes(dpy, gc, 0, LineSolid, CapButt, JoinMiter);

    XSetFillStyle(dpy, gc, FillSolid);

    for (i = 0; i < 5; i++)
	NdcToX(ws, bd, &box[i]);		/* compound-statement macro */

    XDrawLines(dpy, win, gc, box, 5, CoordModeOrigin);

    XFlush(ws->dpy);

    XSetFunction(dpy, gc, GXcopy);
    (void) XgksSIGIO_ON(ws->dpy);

    return OK;
}
