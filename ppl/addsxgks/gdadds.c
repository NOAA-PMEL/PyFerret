/* Extensions to GD libaray */
/* J. Sirott PMEL 1997 */


#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "gif/gd.h"
#include "gif/gdadds.h"
/* absolute value of a */
#define ABS(a)		(((a)<0) ? -(a) : (a))
/* take binary sign of a, either -1, or 1 if >= 0 */
#define SGN(a)		(((a)<0) ? -1 : 1)


static void calc_points(int x1, int y1, int x2, int y2, int width,
			gdPointPtr pts)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    double hypot = sqrt(dx*dx +dy*dy);
    double wsin2t = 0;
    double wcos2t = 0;
    double sinFudge = 0., cosFudge = 0.;
    if (hypot != 0){
      wsin2t = width * dy/hypot/2.;
      wcos2t = width * dx/hypot/2.;
      sinFudge = 0.5 * SGN(wsin2t);
      cosFudge = 0.5 * SGN(wcos2t);
    }
    pts[0].x = x1 - wsin2t - sinFudge;;
    pts[0].y = y1 + wcos2t + cosFudge;
    pts[1].x = x1 + wsin2t + sinFudge;
    pts[1].y = y1 - wcos2t - cosFudge;
    pts[2].x = x2 + wsin2t + sinFudge;
    pts[2].y = y2 - wcos2t - cosFudge;
    pts[3].x = x2 - wsin2t - sinFudge;
    pts[3].y = y2 + wcos2t + cosFudge;
}

typedef struct _StackItem{
  void *next;
  void *data;
} StackItem;

typedef struct _Stack{
  StackItem *item;
} Stack;

static void init(Stack *s)
{
  s->item = 0;
}

static void push(Stack *s, void *data)
{
  StackItem *ns = (StackItem *)umalloc(sizeof(StackItem));
  ns->next = s->item;
  ns->data = data;
  s->item = ns;
}
  
static void *pop(Stack *s)
{
  StackItem *oitem = s->item;
  void *data = oitem ? oitem->data : 0;
  if (oitem){
    s->item = oitem->next;
    free(oitem);
  }
  return data;
}

/*
 * Digital Line Drawing
 * by Paul Heckbert
 * from "Graphics Gems", Academic Press, 1990
 */

/*
 * digline: draw digital line from (x1,y1) to (x2,y2),
 * calling a user-supplied procedure at each pixel.
 * Does no clipping.  Uses Bresenham's algorithm.
 *
 * Paul Heckbert	3 Sep 85
 */

static void digline(int x1, int y1, int x2, int y2,
		    void (*dotproc)(int, int, int, int, void *), void *cd)
{
    int d, x, y, ax, ay, sx, sy, dx, dy;
    int lastdx = 0, lastdy = 0;

    dx = x2-x1;  ax = ABS(dx)<<1;  sx = SGN(dx);
    dy = y2-y1;  ay = ABS(dy)<<1;  sy = SGN(dy);

    x = x1;
    y = y1;
    if (ax>ay) {		/* x dominant */
	d = ay-(ax>>1);
	for (;;) {
	    (*dotproc)(x, y, lastdx, lastdy, cd);
	    if (x==x2) return;
	    if (d>=0) {
		y += sy;
		d -= ax;
	    }
	    x += sx;
	    d += ay;
	    lastdx = sx; lastdy = sy;
	}
    }
    else {			/* y dominant */
	d = ax-(ay>>1);
	for (;;) {
	    (*dotproc)(x, y, lastdx, lastdy, cd);
	    if (y==y2) return;
	    if (d>=0) {
		x += sx;
		d -= ay;
	    }
	    y += sy;
	    d += ax;
	    lastdx = sx; lastdy = sy;
	}
    }
}

typedef struct binfo {
  Stack *s;
} Binfo;

static void storeBoundaries(int x, int y, int dx, int dy, void *cd)
{
  Binfo *info = (Binfo *)cd;
  gdPoint *p;
  if (ABS(dx)== 1 && ABS(dy) == 1){
    int mult = dx * dy;
    p = (gdPoint *)umalloc(sizeof(gdPoint));
    p->x = SGN(mult) < 0 ? -dx : 0;
    p->y = SGN(mult) > 0 ? -dy : 0;
    p->x += x; p->y += y;
    push(info->s, p);
  }
  p = (gdPoint *)umalloc(sizeof(gdPoint));
  p->x = x; p->y = y;
  push(info->s, p);
}

/* Fill a rotatated box by using Bresenham's algorithm to generate */
/* the left and right sides, and then drawing lines between the sides. */
/* Used for generating dashed/dotted lines */

static void  gdImageRotatedBox(gdImagePtr im, gdPointPtr recpts,
			       int num, int color)
{
  Stack left, right, tmp;
  Binfo ileft, iright;
  gdPointPtr first, second;
  int styleStartPos = im->stylePos;

  init(&left); init(&right); init(&tmp);
  ileft.s = &left; iright.s = &right;
  digline(recpts[1].x, recpts[1].y,
	  recpts->x, recpts->y, storeBoundaries, &ileft);
  digline(recpts[3].x, recpts[3].y, recpts[2].x, recpts[2].y,
	  storeBoundaries, &iright);

  /* Reverse order of right stack */
  while((first = (gdPointPtr)pop(iright.s))){
    push(&tmp, first);
  }

  while((first = (gdPointPtr)pop(ileft.s))){
    im->stylePos = styleStartPos;
    second = (gdPointPtr)pop(&tmp);
    assert(second);
    gdImageLine(im, first->x, first->y, second->x, second->y, color);
    free(first);
    free(second);
  }
  
}


/* Draw bevel to connect wide lines */
static void draw_bevel(gdImagePtr im, gdPointPtr recpts,
		       gdPointPtr last, int bcolor)
{
  gdPoint bpt[4];
  if (!last)
    return;
  bpt[0].x = last[2].x; 
  bpt[0].y = last[2].y; 
  bpt[1].x = recpts[1].x; 
  bpt[1].y = recpts[1].y; 
  bpt[2].x = last[3].x; 
  bpt[2].y = last[3].y; 
  bpt[3].x = recpts[0].x; 
  bpt[3].y = recpts[0].y; 
  gdImageFilledPolygon(im, bpt, 4, bcolor);
#if 0
  {
    int i;
    for (i=0; i < 4; ++i){
      fprintf(stderr, "(%d,%d)", bpt[i].x, bpt[i].y);
    }
    fprintf(stderr, "\n");
  }
#endif
}

/* Draw wide lines. Lines are joined with a bevel */

void gdImageWideLines(gdImagePtr im, gdPointPtr pts, int num,
		     int color, int width)
{
  int i;
  Stack s1,s2;
  gdPointPtr recpts = 0;
  gdPointPtr first = pts, second = first + 1;
  gdPoint *last = 0;

  if (width < 1)
    width = 1;

  if (width == 1){
    for (i=1; i < num; ++i,++first,++second){
        gdImageLine(im, first->x, first->y, second->x, second->y, color);
    }
  } else {
    init(&s1); init(&s2);
    for (i=1; i < num; ++i,++first,++second){
      recpts = (gdPointPtr)umalloc(sizeof(gdPoint) * 4);
      calc_points(first->x, first->y, second->x, second->y, width, recpts);
      push(&s1, recpts);
    }

    while((recpts = (gdPointPtr)pop(&s1))){
      push(&s2, recpts);
    }

    while((recpts = (gdPointPtr)pop(&s2))){

      if (color == gdStyled){
	if (im->style[im->stylePos] != gdTransparent){
	  draw_bevel(im, recpts, last, im->style[0]);
	}
	/* Reset style pointer to beginning of the */
	/* current color to avoid slivers */
	if (last){
	  int currStyle = im->style[im->stylePos];
	  int count = 0;
	  for (; count < im->styleLength; ++count){
	    --im->stylePos;
	    im->stylePos %= im->styleLength;
	    if (im->style[im->stylePos] != currStyle)
	      break;
	  }
	  if (count < im->styleLength){
	    ++im->stylePos;
	    im->stylePos %= im->styleLength;
	  }
	}
	
	gdImageRotatedBox(im, recpts, 4, color);
      } else {
	draw_bevel(im, recpts, last, color);
	gdImageFilledPolygon(im, recpts, 4, color);
      }
      free(last); 
      last = recpts;
    }
    if (last)
      free(last);
  }
}

/*
 * Block fill an entire image
 */

void gdImageBlockFill(gdImagePtr image, int color)
{
  int i,j;
  for (i=0; i < image->sx; ++i){
    memset(image->pixels[i], color, image->sy);
  }
}
