/*
 * JC_Map.c
 *
 * Jonathan Callahan
 * Dec 19th 1995
 *
 * This file contains functions which interact with the map for the Ferret GUI.
 *
 */


/* .................... Defines .................... */


#define NUMLATDEGS 180.0
#define NUMLONDEGS 720.0
#define NUMLATPIXELS 216.0
#define NUMLONPIXELS 627.0
#define LONGITUDE 0
#define LATITUDE 1

enum { HNDL_CENTER, HNDL_N, HNDL_NE, HNDL_E, HNDL_SE, HNDL_S, HNDL_SW, HNDL_W, HNDL_NW }
HNDL_type;

enum { TOOL_XYRECT, TOOL_XLINE, TOOL_YLINE, TOOL_POINT, TOOL_NONE }
TOOL_type;

#define DRAG_RECT 111
#define DRAG_XL 222
#define DRAG_YL 333
#define DRAG_PT 444
#define DRAG_CXL 555
#define DRAG_CYL 666


typedef struct {
	int x, y, type;
} JC_Handle;

static XRectangle dataRegionRect;
static XPoint dataRegionPoints[12];
static int gNumHandles=0;
static JC_Handle gHandleList[13];


static XRectangle XYRect;
static XPoint MouseDown;
static int handle_type;
static Boolean gDragInHandle=FALSE;


/* .................... Internal Map Methods .................... */


static void  JC_MapRectangle_BumpAgainstSides( XRectangle *Rect, int x, int y );
static void  JC_MapRectangle_Resize( XRectangle *Rect, int x, int y );
static void  JC_MapRectangle_SnapToGrid( XRectangle *Rect );
static void  JC_Map_Clear(void);
static void  JC_Map_DrawDataRegion(void);
static void  JC_Map_DrawHandle(int x, int y, GC *lgc);
static void  JC_Map_DrawHandle_OldVersion(int x, int y, Boolean storeHandle, int type, GC *lgc);
static void  JC_Map_DrawPoint( XRectangle *Rect );
static void  JC_Map_DrawXYRect( XRectangle *Rect );
static void  JC_Map_DrawXLine( XRectangle *Rect );
static void  JC_Map_DrawYLine( XRectangle *Rect );
static void  JC_Map_DrawCXLine( XRectangle *Rect );
static void  JC_Map_DrawCYLine( XRectangle *Rect );
static float JC_Map_ReturnFloat( int pixel, int lat_or_long );
static int   JC_Map_ReturnHandle(int x, int y);
static int   JC_Map_ReturnPixel( double value, int lat_or_long );
static void  JC_Map_SavePoint( XRectangle *Rect );
static void  JC_Map_SaveXYRect( XRectangle *Rect );
static void  JC_Map_SaveXLine( XRectangle *Rect );
static void  JC_Map_SaveCXLine( XRectangle *Rect );
static void  JC_Map_SaveYLine( XRectangle *Rect );
static void  JC_Map_SaveCYLine( XRectangle *Rect );
static void  JC_Map_StoreHandle( int x, int y, int type );

static Boolean PointInHandle( int inX, int inY, int i );

/* .................... Includes .................... */


/* .................... Map Actions .................... */
/*
 * These functions make up the Action Table associated with
 * mouse clicks in the drawing area.
 */


void JC_Map_Button1Press_Action(Widget wid, XEvent *ev, String *param, Cardinal num)
{
  int x=0, y=0, hdl=0;

  x = ev->xbutton.x;
  x = (x > NUMLONPIXELS) ? NUMLONPIXELS : x;
  x = (x < 0) ? 0 : x;
  y = ev->xbutton.y;
  y = (y > NUMLATPIXELS) ? NUMLATPIXELS : y;;
  y = (y < 0) ? 0 : y;


  if ( hdl=JC_Map_ReturnHandle(x, y) ) {

    gDragInHandle = TRUE;
    handle_type = gHandleList[hdl].type;

  } else {

    gDragInHandle = FALSE;
    MouseDown.x = x;
    MouseDown.y = y;

  }
}


void JC_Map_Motion1Notify_Action(Widget wid, XEvent *ev, String *param, Cardinal num)
{
  JC_Region *R_ptr=&GLOBAL_Region;
  int x=0, y=0;

  if ( toolMode == xy ) tool_type = TOOL_XYRECT;
  else if ( toolMode == xl ) tool_type = TOOL_XLINE;
  else if ( toolMode == yl ) tool_type = TOOL_YLINE;
  else if ( toolMode == pt ) tool_type = TOOL_POINT;
  else if ( toolMode == no ) tool_type = TOOL_NONE;

  x = ev->xmotion.x;
  y = ev->xmotion.y;
  x = (x > NUMLONPIXELS) ? NUMLONPIXELS : x;
  x = (x < 0) ? 0 : x;
  y = (y > NUMLATPIXELS) ? NUMLATPIXELS : y;
  y = (y < 0) ? 0 : y;
     
  JC_Map_Clear();

  if ( gDragInHandle ) {

    if ( handle_type == HNDL_CENTER )
      JC_MapRectangle_BumpAgainstSides(&XYRect, x, y);
    else
      JC_MapRectangle_Resize(&XYRect, x, y);
    /*
      JC_MapRectangle_SnapToGrid(&XYRect);
      */
    if ( tool_type == TOOL_XYRECT ) JC_Map_DrawXYRect(&XYRect);
    else if ( tool_type == TOOL_XLINE ) JC_Map_DrawXLine(&XYRect);
    else if ( tool_type == TOOL_YLINE ) JC_Map_DrawYLine(&XYRect);
    else if ( tool_type == TOOL_POINT ) JC_Map_DrawPoint(&XYRect);
    else if ( tool_type == TOOL_NONE ) ;
    else
      fprintf(stderr, "ERROR in JC_Map.c: JC_Map_Motion1Notify_Action(): tool_type = %d\n", tool_type);

  } else /* We are drawing the tool from scratch. */ {

    if ( R_ptr->span[X_AXIS].needs_lo_hi_displayed_in_GUI ) {
      if ( x > MouseDown.x ) {
	XYRect.width = x - MouseDown.x;
	XYRect.x = MouseDown.x + XYRect.width/2;
      } else {
	XYRect.width = MouseDown.x - x;
	XYRect.x = x + XYRect.width/2;
      }
    } else {
      XYRect.x = MouseDown.x;
      XYRect.width = 1;
    }

    if ( R_ptr->span[Y_AXIS].needs_lo_hi_displayed_in_GUI ) {
      if ( y > MouseDown.y ) {
	XYRect.height = y - MouseDown.y;
	XYRect.y = MouseDown.y + XYRect.height/2;
      } else {
	XYRect.height = MouseDown.y - y;
	XYRect.y = y + XYRect.height/2;
      }
    } else {
      XYRect.y = MouseDown.y;
      XYRect.height = 1;
    }
    /*  
	JC_MapRectangle_SnapToGrid(&XYRect);
	*/
    if ( tool_type == TOOL_XYRECT ) JC_Map_DrawXYRect(&XYRect);
    else if ( tool_type == TOOL_XLINE ) JC_Map_DrawXLine(&XYRect);
    else if ( tool_type == TOOL_YLINE ) JC_Map_DrawYLine(&XYRect);
    else if ( tool_type == TOOL_POINT ) JC_Map_DrawPoint(&XYRect);
    else if ( tool_type == TOOL_NONE ) ;
    else
      fprintf(stderr, "ERROR in JC_Map.c: JC_Map_Motion1Notify_Action(): tool_type = %d\n", tool_type);
 }

  if ( R_ptr->span[X_AXIS].is_compressed_in_GUI )
    JC_Map_DrawCXLine(&XYRect);
  if ( R_ptr->span[Y_AXIS].is_compressed_in_GUI )
    JC_Map_DrawCYLine(&XYRect);
     
}


void JC_Map_Button1Release_Action(Widget wid, XEvent *ev, String *param, Cardinal num)
{
  JC_Region *R_ptr=&GLOBAL_Region;
  JC_Variable *V_ptr=&GLOBAL_Variable;

  float lon[3]={0,0,0}, lat[3]={0,0,0};


  if ( toolMode == xy ) tool_type = TOOL_XYRECT;
  else if ( toolMode == xl ) tool_type = TOOL_XLINE;
  else if ( toolMode == yl ) tool_type = TOOL_YLINE;
  else if ( toolMode == pt ) tool_type = TOOL_POINT;
  else if ( toolMode == no ) tool_type = TOOL_NONE;

  gNumHandles = 0;
     
  switch (tool_type) {
	  
  case TOOL_XYRECT:
    JC_Map_SaveXYRect(&XYRect);
    break;
	       
  case TOOL_XLINE:
    JC_Map_SaveXLine(&XYRect);
    break;

  case TOOL_YLINE:
    JC_Map_SaveYLine(&XYRect);
    break;

  case TOOL_POINT:
    JC_Map_SavePoint(&XYRect);
    break;

  case TOOL_NONE:
    JC_Map_SavePoint(&XYRect);
    break;

  default:
    fprintf(stderr, "ERROR in JC_Map.c: JC_Map_Motion1Notify_Action(): tool_type = %d\n", tool_type);
    break;

  }
     
  if ( R_ptr->span[X_AXIS].is_compressed_in_GUI )
    JC_Map_SaveCXLine(&XYRect);
  if ( R_ptr->span[Y_AXIS].is_compressed_in_GUI )
    JC_Map_SaveCYLine(&XYRect);
     
  /*
    lon[LO] = JC_Map_ReturnFloat(XYRect.x - XYRect.width/2, LONGITUDE);
    lon[HI] = JC_Map_ReturnFloat(XYRect.x - XYRect.width/2 + XYRect.width, LONGITUDE);
    *
    lon[PT] = JC_Map_ReturnFloat(XYRect.x, LONGITUDE);
    *
    lat[LO] = JC_Map_ReturnFloat(XYRect.y - XYRect.height/2 + XYRect.height, LATITUDE);
    lat[HI] = JC_Map_ReturnFloat(XYRect.y - XYRect.height/2, LATITUDE);
    *
    lat[PT] = JC_Map_ReturnFloat(XYRect.y, LATITUDE);
    *
    lon[LO] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[X_AXIS]), lon[LO]);
    lon[HI] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[X_AXIS]), lon[HI]);
    lat[LO] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[Y_AXIS]), lat[LO]);
    lat[HI] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[Y_AXIS]), lat[HI]);
    lon[PT] = lon[HI] - lon[LO];
    lat[PT] = lat[HI] - lat[LO];
    */

  lon[LO] = JC_Map_ReturnFloat(XYRect.x - XYRect.width/2, LONGITUDE);
  lon[HI] = JC_Map_ReturnFloat(XYRect.x - XYRect.width/2 + XYRect.width, LONGITUDE);
  lon[PT] = JC_Map_ReturnFloat(XYRect.x, LONGITUDE);
  lat[LO] = JC_Map_ReturnFloat(XYRect.y - XYRect.height/2 + XYRect.height, LATITUDE);
  lat[HI] = JC_Map_ReturnFloat(XYRect.y - XYRect.height/2, LATITUDE);
  lat[PT] = JC_Map_ReturnFloat(XYRect.y, LATITUDE);
 

  JC_Region_NewFramer( R_ptr, lat, lon );
  JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[X_AXIS]) );
  JC_GeometryInterfaceLine_NewSpan( &(R_ptr->span[Y_AXIS]) );
     
}

/* .................... Map Methods .................... */


void JC_Map_NewVariable( JC_Variable *V_ptr )
{
  double value=0;
  int top=0, bottom=0, left=0, right=0;

  left   = JC_Map_ReturnPixel(V_ptr->axis[X_AXIS].ww[LO], LONGITUDE);
  left   = (left < 1) ? 1 : left;
  right  = JC_Map_ReturnPixel(V_ptr->axis[X_AXIS].ww[HI], LONGITUDE);
  right  = (right > NUMLONPIXELS-1) ? NUMLONPIXELS-1 : right;
  top    = JC_Map_ReturnPixel(V_ptr->axis[Y_AXIS].ww[HI], LATITUDE);
  top    = (top < 1) ? 1 : top;
  bottom = JC_Map_ReturnPixel(V_ptr->axis[Y_AXIS].ww[LO], LATITUDE);
  bottom = (bottom > NUMLATPIXELS-1) ? NUMLATPIXELS-1 : bottom;

  dataRegionRect.x = left;
  dataRegionRect.y = top;
  dataRegionRect.width = (right-left);
  dataRegionRect.height = (bottom-top);


  dataRegionPoints[0].x = 0;
  dataRegionPoints[0].y = 0;
     
  dataRegionPoints[1].x = NUMLONPIXELS;
  dataRegionPoints[1].y = 0;

  dataRegionPoints[2].x = NUMLONPIXELS;
  dataRegionPoints[2].y = NUMLATPIXELS;

  dataRegionPoints[3].x = 0;
  dataRegionPoints[3].y = NUMLATPIXELS;

  dataRegionPoints[4].x = 0;
  dataRegionPoints[4].y = 0;

  value = V_ptr->axis[X_AXIS].ww[LO];
  if ( V_ptr->axis[X_AXIS].is_modulo )
    dataRegionPoints[5].x = 0;
  else
    dataRegionPoints[5].x = JC_Map_ReturnPixel(value, LONGITUDE) + 1;

  dataRegionPoints[5].y = 0;


  value = V_ptr->axis[X_AXIS].ww[LO];
  if ( V_ptr->axis[X_AXIS].is_modulo )
    dataRegionPoints[6].x = 0;
  else
    dataRegionPoints[6].x =  JC_Map_ReturnPixel(value, LONGITUDE) + 1;
  value = V_ptr->axis[Y_AXIS].ww[LO];

  dataRegionPoints[6].y = JC_Map_ReturnPixel(value, LATITUDE) + 0/* 1 */;


  value = V_ptr->axis[X_AXIS].ww[HI];
  if ( V_ptr->axis[X_AXIS].is_modulo )
    dataRegionPoints[7].x = NUMLONPIXELS;
  else
    dataRegionPoints[7].x = JC_Map_ReturnPixel(value, LONGITUDE) + 1;

  value = V_ptr->axis[Y_AXIS].ww[LO];
  dataRegionPoints[7].y = JC_Map_ReturnPixel(value, LATITUDE) + 0/* 1 */;


  value = V_ptr->axis[X_AXIS].ww[HI];
  if ( V_ptr->axis[X_AXIS].is_modulo )
    dataRegionPoints[8].x = NUMLONPIXELS;
  else
    dataRegionPoints[8].x = JC_Map_ReturnPixel(value, LONGITUDE) + 1;

  value = V_ptr->axis[Y_AXIS].ww[HI];
  dataRegionPoints[8].y = JC_Map_ReturnPixel(value, LATITUDE) + 1;


  value = V_ptr->axis[X_AXIS].ww[LO];
  if ( V_ptr->axis[X_AXIS].is_modulo )
    dataRegionPoints[9].x = 0;
  else
    dataRegionPoints[9].x = JC_Map_ReturnPixel(value, LONGITUDE) + 1;

  value = V_ptr->axis[Y_AXIS].ww[HI];
  dataRegionPoints[9].y = JC_Map_ReturnPixel(value, LATITUDE) + 1;


  value = V_ptr->axis[X_AXIS].ww[LO];
  if ( V_ptr->axis[X_AXIS].is_modulo )
    dataRegionPoints[10].x = 0;
  else
    dataRegionPoints[10].x =  JC_Map_ReturnPixel(value, LONGITUDE) + 1;

  dataRegionPoints[10].y = 0;
     

  dataRegionPoints[11].x = 0;
  dataRegionPoints[11].y = 0;

}


static void JC_Map_DrawDataRegion( void )
{
  XFillPolygon(XtDisplay(drawingArea1), 
	       XtWindow(drawingArea1), drGc, 
	       dataRegionPoints, 12, Complex, CoordModeOrigin);
}


/*
 * This next routine copied from O'Reilly "Motif Programming Manual V6", p. 357
 */
void JC_Map_SetToolColor( XtPointer color_name )
{
  String color = (String) color_name;
  Display *dpy = XtDisplay (drawingArea1);
  Colormap cmap = DefaultColormapOfScreen (XtScreen (drawingArea1));
  XColor col, unused;

  if (!XAllocNamedColor (dpy, cmap, color, &col, &unused)) {
    char buf[32];
    sprintf (buf, "JC_Map_SetToolColor: Can't alloc %s", color);
    XtWarning (buf);
    return;
  }
  XSetForeground (dpy, gc, col.pixel);
}


void JC_Map_NewRegion( JC_Region *R_ptr )
{
  float lat[2]={0,0}, lon[2]={0,0};

  JC_Map_Clear();
  gNumHandles = 0;

	
  if ( R_ptr->span[X_AXIS].needs_lo_hi_displayed_in_GUI ) {
    lon[LO] = R_ptr->span[X_AXIS].ww[LO];
    lon[HI] = R_ptr->span[X_AXIS].ww[HI];
  } else {
    lon[LO] = R_ptr->span[X_AXIS].ww[PT];
    lon[HI] = R_ptr->span[X_AXIS].ww[PT];
  }
	
  if ( R_ptr->span[Y_AXIS].needs_lo_hi_displayed_in_GUI ) {
    lat[LO] = R_ptr->span[Y_AXIS].ww[LO];
    lat[HI] = R_ptr->span[Y_AXIS].ww[HI];
  } else {
    lat[LO] = R_ptr->span[Y_AXIS].ww[PT];
    lat[HI] = R_ptr->span[Y_AXIS].ww[PT];
  }
	
  XYRect.width = JC_Map_ReturnPixel(lon[HI], LONGITUDE) -  JC_Map_ReturnPixel(lon[LO], LONGITUDE);
  if (XYRect.width <= 0 || XYRect.width > NUMLONPIXELS) XYRect.width = 1;
  XYRect.height = JC_Map_ReturnPixel(lat[LO], LATITUDE) -  JC_Map_ReturnPixel(lat[HI], LATITUDE);
  if (XYRect.height <= 0 || XYRect.height > NUMLATPIXELS) XYRect.height = 1;
  XYRect.x = JC_Map_ReturnPixel(lon[LO],LONGITUDE) + XYRect.width/2;
  XYRect.y = JC_Map_ReturnPixel(lat[HI],LATITUDE) + XYRect.height/2;

  if ( R_ptr->span[X_AXIS].ss[LO] != IRRELEVANT_AXIS  && R_ptr->span[Y_AXIS].ss[LO] != IRRELEVANT_AXIS ) {

    switch (R_ptr->geometry) {

    case GEOM_POINT:
    case GEOM_Z:
    case GEOM_T:
    case GEOM_ZT:
      JC_Map_DrawPoint(&XYRect);
      JC_Map_SavePoint(&XYRect);
      toolMode = pt;
      mapMode = pt;
      break;

    case GEOM_X:
    case GEOM_XZ:
    case GEOM_XT:
    case GEOM_XZT:
      JC_Map_DrawXLine(&XYRect);
      JC_Map_SaveXLine(&XYRect);
      toolMode = xl;
      mapMode = xl;
      break;

    case GEOM_Y:
    case GEOM_YZ:
    case GEOM_YT:
    case GEOM_YZT:
      JC_Map_DrawYLine(&XYRect);
      JC_Map_SaveYLine(&XYRect);
      toolMode = yl;
      mapMode = yl;
      break;

    case GEOM_XY:
    case GEOM_XYZ:
    case GEOM_XYT:
    case GEOM_XYZT:
      JC_Map_DrawXYRect(&XYRect);
      JC_Map_SaveXYRect(&XYRect);
      toolMode = xy;
      mapMode = xy;
      break;

    }

  } else {

    JC_Map_SavePoint(&XYRect);
    toolMode = no;
    mapMode = no;

  }
	
  if ( R_ptr->span[X_AXIS].is_compressed_in_GUI ) {
    JC_Map_DrawCXLine(&XYRect);
    JC_Map_SaveCXLine(&XYRect);
  }
  if ( R_ptr->span[Y_AXIS].is_compressed_in_GUI ) {
    JC_Map_DrawCYLine(&XYRect);
    JC_Map_SaveCYLine(&XYRect);
  }
}


static float JC_Map_ReturnFloat( int pixel, int lat_or_long )
{

  if ( lat_or_long == LONGITUDE ) { 

    return ( (float)pixel * (NUMLONDEGS/NUMLONPIXELS) );

  } else if ( lat_or_long == LATITUDE )

    return ( -(float)pixel * (NUMLATDEGS/NUMLATPIXELS) + 90.0 );

  else

    fprintf(stderr, "ERROR in JC_Map.c: JC_Map_ReturnFloat(): lat_or_long = %d.\n", lat_or_long);

}


static int JC_Map_ReturnPixel( double value, int lat_or_long )
{
  if ( lat_or_long == LONGITUDE ) { 

    if ( value >= 0.0 )
      return (int)(value * NUMLONPIXELS/NUMLONDEGS);
    else 
      return (int)((value+360) * NUMLONPIXELS/NUMLONDEGS);

  } else if ( lat_or_long == LATITUDE )

    return (int) ( -((value-90.0) * NUMLATPIXELS/NUMLATDEGS) );

  else

    fprintf(stderr, "ERROR in JC_Map.c: JC_Map_ReturnPixel(): lat_or_long = %d.\n", lat_or_long);

}


static void JC_Map_Clear()
{
  XClearWindow(XtDisplay(UxGetWidget(drawingArea1)),
	       XtWindow(UxGetWidget(drawingArea1)));
  JC_Map_DrawDataRegion();
}

static void JC_Map_DrawXYRect( XRectangle *Rect )
{
  GC lgc;
  int left   = Rect->x - Rect->width/2;
  int right  = left + Rect->width;
  int top    = Rect->y - Rect->height/2;
  int bottom = top + Rect->height;
     
  XtVaGetValues(UxGetWidget(drawingArea1),
		XmNuserData, &lgc,
		NULL);
  XDrawRectangle(XtDisplay(UxGetWidget(drawingArea1)), 
		 XtWindow(UxGetWidget(drawingArea1)), lgc,
		 left, top, Rect->width, Rect->height);

  JC_Map_DrawHandle(left,    top,     &lgc); /* NW */
  JC_Map_DrawHandle(right,   top,     &lgc); /* NE */
  JC_Map_DrawHandle(right,   bottom,  &lgc); /* SE */
  JC_Map_DrawHandle(left,    bottom,  &lgc); /* SW */

  JC_Map_DrawHandle(Rect->x, Rect->y, &lgc); /* CENTER */
  JC_Map_DrawHandle(Rect->x, top,     &lgc); /* N */
  JC_Map_DrawHandle(right,   Rect->y, &lgc); /* E */
  JC_Map_DrawHandle(Rect->x, bottom,  &lgc); /* S */
  JC_Map_DrawHandle(left,    Rect->y, &lgc); /* W */
}

static void JC_Map_DrawXLine( XRectangle *Rect )
{
  GC lgc;
  int left   = Rect->x - Rect->width/2;
  int right  = left + Rect->width;

  XtVaGetValues(UxGetWidget(drawingArea1),
		XmNuserData, &lgc,
		NULL);
  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    left,  Rect->y,	right, Rect->y);

  JC_Map_DrawHandle(left,    Rect->y, &lgc);/* E */
  JC_Map_DrawHandle(Rect->x, Rect->y, &lgc);/* CENTER */
  JC_Map_DrawHandle(right,   Rect->y, &lgc);/* W */
}

static void JC_Map_DrawYLine( XRectangle *Rect )
{
  GC lgc;
  int top    = Rect->y - Rect->height/2;
  int bottom = top + Rect->height;

  XtVaGetValues(UxGetWidget(drawingArea1),
		XmNuserData, &lgc,
		NULL);
  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x, top, Rect->x, bottom);

  JC_Map_DrawHandle(Rect->x, top,     &lgc);/* N */
  JC_Map_DrawHandle(Rect->x, Rect->y, &lgc);/* CENTER */
  JC_Map_DrawHandle(Rect->x, bottom,  &lgc);/* S */
}

static void JC_Map_DrawCXLine( XRectangle *Rect )
{
  GC lgc;
  int left   = Rect->x - Rect->width/2;
  int right  = left + Rect->width;

  XtVaGetValues(UxGetWidget(drawingArea1),
		XmNuserData, &lgc,
		NULL);
  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    left, Rect->y, right, Rect->y);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-9, Rect->y-4,
	    Rect->x-2, Rect->y);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-9, Rect->y+4,
	    Rect->x-2, Rect->y);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-9, Rect->y-4,
	    Rect->x-9, Rect->y+4);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x+9, Rect->y-4,
	    Rect->x+2, Rect->y);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x+9, Rect->y+4,
	    Rect->x+2, Rect->y);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x+9, Rect->y-4,
	    Rect->x+9, Rect->y+4);

  JC_Map_DrawHandle(left,   Rect->y, &lgc);
  JC_Map_DrawHandle(right,  Rect->y, &lgc);

}

static void JC_Map_DrawCYLine( XRectangle *Rect )
{
  GC lgc;
  int top    = Rect->y - Rect->height/2;
  int bottom = top + Rect->height;

  XtVaGetValues(UxGetWidget(drawingArea1),
		XmNuserData, &lgc,
		NULL);
  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x, top, Rect->x, bottom);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-4, Rect->y-9,
	    Rect->x, Rect->y-9);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-4, Rect->y-9,
	    Rect->x, Rect->y-2);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x+4, Rect->y-9,
	    Rect->x, Rect->y-2);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-4, Rect->y+9,
	    Rect->x+4, Rect->y+9);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-4, Rect->y+9,
	    Rect->x, Rect->y+2);

  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x+4, Rect->y+9,
	    Rect->x, Rect->y+2);

  JC_Map_DrawHandle(Rect->x, top,    &lgc);
  JC_Map_DrawHandle(Rect->x, bottom, &lgc);
}

static void JC_Map_DrawPoint( XRectangle *Rect )
{
  GC lgc;

  XtVaGetValues(UxGetWidget(drawingArea1),
		XmNuserData, &lgc,
		NULL);
  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x-10, Rect->y,
	    Rect->x+10, Rect->y);
  XDrawLine(XtDisplay(UxGetWidget(drawingArea1)), 
	    XtWindow(UxGetWidget(drawingArea1)), lgc,
	    Rect->x, Rect->y-10,
	    Rect->x, Rect->y+10);

  JC_Map_DrawHandle(Rect->x, Rect->y, &lgc);
}


static void JC_Map_DrawHandle( int x, int y, GC *lgc)
{
  XFillRectangle(XtDisplay(drawingArea1), 
		 XtWindow(drawingArea1), *lgc,
		 x-2, y-2, 4, 4);
}

static void JC_Map_SaveXYRect( XRectangle *Rect )
{
  int left   = Rect->x - Rect->width/2;
  int right  = left + Rect->width;
  int top    = Rect->y - Rect->height/2;
  int bottom = top + Rect->height;
 
  JC_Map_StoreHandle(left,    top,     HNDL_NW);
  JC_Map_StoreHandle(right,   top,     HNDL_NE);
  JC_Map_StoreHandle(right,   bottom,  HNDL_SE);
  JC_Map_StoreHandle(left,    bottom,  HNDL_SW);

  JC_Map_StoreHandle(Rect->x, Rect->y, HNDL_CENTER);
  JC_Map_StoreHandle(Rect->x, top,     HNDL_N);
  JC_Map_StoreHandle(right,   Rect->y, HNDL_E);
  JC_Map_StoreHandle(Rect->x, bottom,  HNDL_S);
  JC_Map_StoreHandle(left,    Rect->y, HNDL_W);
}

static void JC_Map_SaveXLine( XRectangle *Rect )
{
  int left   = Rect->x - Rect->width/2;
  int right  = left + Rect->width;

  JC_Map_StoreHandle(left,    Rect->y, HNDL_W);
  JC_Map_StoreHandle(Rect->x, Rect->y, HNDL_CENTER);
  JC_Map_StoreHandle(right,   Rect->y, HNDL_E);
}

static void JC_Map_SaveYLine( XRectangle *Rect )
{
  int top    = Rect->y - Rect->height/2;
  int bottom = top + Rect->height;
 
  JC_Map_StoreHandle(Rect->x, top,     HNDL_N);
  JC_Map_StoreHandle(Rect->x, Rect->y, HNDL_CENTER);
  JC_Map_StoreHandle(Rect->x, bottom,  HNDL_S);
}

static void JC_Map_SaveCXLine( XRectangle *Rect )
{
  int left   = Rect->x - Rect->width/2;
  int right  = left + Rect->width;

  JC_Map_StoreHandle(left,    Rect->y, HNDL_W);
  JC_Map_StoreHandle(right,   Rect->y, HNDL_E);
}

static void JC_Map_SaveCYLine( XRectangle *Rect )
{
  int top    = Rect->y - Rect->height/2;
  int bottom = top + Rect->height;
 
  JC_Map_StoreHandle(Rect->x, top,     HNDL_N);
  JC_Map_StoreHandle(Rect->x, bottom,  HNDL_S);
}

static void JC_Map_SavePoint( XRectangle *Rect )
{
  JC_Map_StoreHandle(Rect->x, Rect->y, HNDL_CENTER);
}

static void JC_Map_StoreHandle( int x, int y, int type )
{
  gNumHandles++;
  gHandleList[gNumHandles].x = x;
  gHandleList[gNumHandles].y = y;
  gHandleList[gNumHandles].type = type;
}

static int JC_Map_ReturnHandle( int inX, int inY )
{
  int i;

  for (i=1; i<=gNumHandles; i++) {
    if (PointInHandle(inX, inY, i))
      return i;
  }
  return 0;
}

static Boolean PointInHandle( int inX, int inY, int i )
{

  if (inX < gHandleList[i].x-3) return FALSE;
  if (inX > gHandleList[i].x+3) return FALSE;
  if (inY < gHandleList[i].y-3) return FALSE;
  if (inY > gHandleList[i].y+3) return FALSE;
  return TRUE;
}

void JC_Map_Show( void )
{
  XtManageChild(UxGetWidget(frame_map));
  XtMapWidget(frame_map);
}

void JC_Map_Hide( void )
{
  XtUnmapWidget(frame_map);
  XtUnmanageChild(UxGetWidget(frame_map));
}


/*
 * This function accepts an XRectangle pointer where x,y define the center of
 * the rectangle and the width and height define the total width and total height.
 *
 * This function tests to see if the rectangle will bump against the sides of the
 * map and returns a rectangle with a new x,y if necessary to avoid bumping.
 */
static void JC_MapRectangle_BumpAgainstSides( XRectangle *Rect, int x, int y )
{
     
  char *handle_name[9]={"CENTER", "N", "NE", "E", "SE", "S", "SW", "W", "NW"};

  int left   = x - Rect->width/2;
  int right  = left + Rect->width;
  int top    = y - Rect->height/2;
  int bottom = top + Rect->height;
     
  if ( left <= 1 )
    Rect->x = 1 + Rect->width/2;
     
  else if ( right > NUMLONPIXELS - 1)
    Rect->x = NUMLONPIXELS - 1 - Rect->width + Rect->width/2;
     
  else
    Rect->x = x;
     

  if ( top <= 1 )
    Rect->y = 1 + Rect->height/2;
     
  else if ( bottom > NUMLATPIXELS - 1 )
    Rect->y = NUMLATPIXELS - 1 - Rect->height + Rect->height/2;
     
  else
    Rect->y = y;
     
    
}


static void JC_MapRectangle_Resize( XRectangle *Rect, int x, int y )
{

  int old_left   = Rect->x - Rect->width/2;
  int old_right  = old_left + Rect->width;
  int old_top    = Rect->y - Rect->height/2;
  int old_bottom = old_top + Rect->height;
  int left=old_left, right=old_right, top=old_top, bottom=old_bottom;
  Boolean handle_overtook_opposite_side=FALSE;

  /*
   * - First deal with the new X:
   */
  if ( x > old_right ) {
    switch (handle_type) {
    case HNDL_NE:
    case HNDL_E:
    case HNDL_SE:
      left = old_left;
      right = x;
      break;
    case HNDL_NW:
    case HNDL_W:
    case HNDL_SW:
      left = old_right;
      right = x;
      handle_overtook_opposite_side = TRUE;
      break;
    case HNDL_N:
    case HNDL_S:
      left = old_left;
      right = old_right;
      break;
    }
  } else if ( x < old_left ) {
    switch (handle_type) {
    case HNDL_NE:
    case HNDL_E:
    case HNDL_SE:
      left = x;
      right = old_left;
      handle_overtook_opposite_side = TRUE;
      break;
    case HNDL_NW:
    case HNDL_W:
    case HNDL_SW:
      left = x;
      right = old_right;
      break;
    case HNDL_N:
    case HNDL_S:
      left = old_left;
      right = old_right;
      break;
    }
  } else {
    switch (handle_type) {
    case HNDL_NE:
    case HNDL_E:
    case HNDL_SE:
      left = old_left;
      right = x;
      break;
    case HNDL_NW:
    case HNDL_W:
    case HNDL_SW:
      left = x;
      right = old_right;
      break;
    case HNDL_N:
    case HNDL_S:
      left = old_left;
      right = old_right;
      break;
    }
  }

  if ( handle_overtook_opposite_side ) {
    if ( handle_type == HNDL_NE ) handle_type = HNDL_NW;
    else if ( handle_type == HNDL_E  ) handle_type = HNDL_W;
    else if ( handle_type == HNDL_SE ) handle_type = HNDL_SW;
    else if ( handle_type == HNDL_NW ) handle_type = HNDL_NE;
    else if ( handle_type == HNDL_W  ) handle_type = HNDL_E;
    else if ( handle_type == HNDL_SW ) handle_type = HNDL_SE;
  }

  handle_overtook_opposite_side = FALSE;

  /*
   * Now deal with the new Y:
   */
  if ( y > old_bottom ) {
    switch (handle_type) {
    case HNDL_NW:
    case HNDL_N:
    case HNDL_NE:
      top = old_bottom;
      bottom = y;
      handle_overtook_opposite_side = TRUE;
      break;
    case HNDL_SW:
    case HNDL_S:
    case HNDL_SE:
      top = old_top;
      bottom = y;
      break;
    case HNDL_E:
    case HNDL_W:
      top = old_top;
      bottom = old_bottom;
      break;
    }
  } else if ( y < old_top ) {
    switch (handle_type) {
    case HNDL_NW:
    case HNDL_N:
    case HNDL_NE:
      top = y;
      bottom = old_bottom;
      break;
    case HNDL_SW:
    case HNDL_S:
    case HNDL_SE:
      top = y;
      bottom = old_top;
      handle_overtook_opposite_side = TRUE;
      break;
    case HNDL_E:
    case HNDL_W:
      top = old_top;
      bottom = old_bottom;
      break;
    }
  } else {
    switch (handle_type) {
    case HNDL_NW:
    case HNDL_N:
    case HNDL_NE:
      top = y;
      bottom = old_bottom;
      break;
    case HNDL_SW:
    case HNDL_S:
    case HNDL_SE:
      top = old_top;
      bottom = y;
      break;
    case HNDL_E:
    case HNDL_W:
      top = old_top;
      bottom = old_bottom;
      break;
    }
  }

  if ( handle_overtook_opposite_side ) {
    if ( handle_type == HNDL_NW ) handle_type = HNDL_SW;
    else if ( handle_type == HNDL_N  ) handle_type = HNDL_S;
    else if ( handle_type == HNDL_NE ) handle_type = HNDL_SE;
    else if ( handle_type == HNDL_SW ) handle_type = HNDL_NW;
    else if ( handle_type == HNDL_S  ) handle_type = HNDL_N;
    else if ( handle_type == HNDL_SE ) handle_type = HNDL_NE;
  }

  /*
   * Now create the new rectangle.
   */
  Rect->width = right - left;
  Rect->height = bottom - top;
  Rect->x = left + Rect->width/2;
  Rect->y = top + Rect->height/2;

}


/*
 * This routine alters the MapRectangle so that the bounding pixles lie on the
 * pixels nearest a valid grid point as determined by the values on the JC_Axis.
 *
 * This has the effect of "snapping to grid" where the grid is the grid of the
 * data values.
 */
static void JC_MapRectangle_SnapToGrid( XRectangle *Rect )
{

  JC_Variable *V_ptr=&GLOBAL_Variable;

  int old_left   = Rect->x - Rect->width/2;
  int old_right  = old_left + Rect->width;
  int old_top    = Rect->y - Rect->height/2;
  int old_bottom = old_top + Rect->height;
  int left=old_left, right=old_right, top=old_top, bottom=old_bottom;

  float old_lon[3]={0,0,0}, old_lat[3]={0,0,0};
  float lon[3]={0,0,0}, lat[3]={0,0,0};

  /*
   * First, convert the current pixels to floats.
   * Then, get the nearest axis values.
   * Then, convert those back to the nearest pixel values.
   */

  old_lon[LO] = JC_Map_ReturnFloat(left, LONGITUDE);
  old_lon[HI] = JC_Map_ReturnFloat(right, LONGITUDE);
  old_lat[LO] = JC_Map_ReturnFloat(bottom, LATITUDE);
  old_lat[HI] = JC_Map_ReturnFloat(top, LATITUDE);

  lon[LO] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[X_AXIS]), old_lon[LO]);
  lon[HI] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[X_AXIS]), old_lon[HI]);
  lat[LO] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[Y_AXIS]), old_lat[LO]);
  lat[HI] = JC_Axis_ReturnNearestMidpoint(&(V_ptr->axis[Y_AXIS]), old_lat[HI]);
  
  left   = JC_Map_ReturnPixel((double)lon[LO], LONGITUDE);
  left   = (left < 1) ? 1 : left;
  right  = JC_Map_ReturnPixel((double)lon[HI], LONGITUDE);
  right  = (right > NUMLONPIXELS-1) ? NUMLONPIXELS-1 : right;
  top    = JC_Map_ReturnPixel((double)lat[HI], LATITUDE);
  top    = (top < 1) ? 1 : top;
  bottom = JC_Map_ReturnPixel((double)lat[LO], LATITUDE);
  bottom = (bottom > NUMLATPIXELS-1) ? NUMLATPIXELS-1 : bottom;
  
  /*
   * Now create the new rectangle.
   */
  Rect->width = right - left;
  Rect->height = bottom - top;
  Rect->x = left + Rect->width/2;
  Rect->y = top + Rect->height/2;

}
