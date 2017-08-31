#ifndef XGKS_METAFILE_H
#define XGKS_METAFILE_H

/*
 * GKSM item identification numbers:
 */
typedef enum gksm_item_id {
    GKSM_CGM_ELEMENT			= -2,
    GKSM_UNKNOWN_ITEM			= -1,
    GKSM_END_ITEM			= 0,
    GKSM_CLEAR_WORKSTATION,
    GKSM_REDRAW_ALL_SEGMENTS_ON_WORKSTATION,
    GKSM_UPDATE_WORKSTATION,
    GKSM_DEFERRAL_STATE,
    GKSM_MESSAGE,
    GKSM_ESCAPE,
    GKSM_POLYLINE			= 11,
    GKSM_POLYMARKER,
    GKSM_TEXT,
    GKSM_FILL_AREA,
    GKSM_CELLARRAY,
    GKSM_GENERALIZED_DRAWING_PRIMITIVE,
    GKSM_POLYLINE_INDEX			= 21,
    GKSM_LINETYPE,
    GKSM_LINEWIDTH_SCALE_FACTOR,
    GKSM_POLYLINE_COLOUR_INDEX,
    GKSM_POLYMARKER_INDEX,
    GKSM_MARKER_TYPE,
    GKSM_MARKER_SIZE_SCALE_FACTOR,
    GKSM_POLYMARKER_COLOUR_INDEX,
    GKSM_TEXT_INDEX,
    GKSM_TEXT_FONT_AND_PRECISION,
    GKSM_CHARACTER_EXPANSION_FACTOR,
    GKSM_CHARACTER_SPACING,
    GKSM_TEXT_COLOUR_INDEX,
    GKSM_CHARACTER_VECTORS,
    GKSM_TEXT_PATH,
    GKSM_TEXT_ALIGNMENT,
    GKSM_FILL_AREA_INDEX,
    GKSM_FILL_AREA_INTERIOR_STYLE,
    GKSM_FILL_AREA_STYLE_INDEX,
    GKSM_FILL_AREA_COLOUR_INDEX,
    GKSM_PATTERN_SIZE,
    GKSM_PATTERN_REFERENCE_POINT,
    GKSM_ASPECT_SOURCE_FLAGS,
    GKSM_PICK_IDENTIFIER,
    GKSM_POLYLINE_REPRESENTATION	= 51,
    GKSM_POLYMARKER_REPRESENTATION,
    GKSM_TEXT_REPRESENTATION,
    GKSM_FILL_AREA_REPRESENTATION,
    GKSM_PATTERN_REPRESENTATION,
    GKSM_COLOUR_REPRESENTATION,
    GKSM_CLIPPING_RECTANGLE		= 61,
    GKSM_WORKSTATION_WINDOW		= 71,
    GKSM_WORKSTATION_VIEWPORT,
    GKSM_CREATE_SEGMENT			= 81,
    GKSM_CLOSE_SEGMENT,
    GKSM_RENAME_SEGMENT,
    GKSM_DELETE_SEGMENT,
    GKSM_SET_SEGMENT_TRANSFORMATION	= 91,
    GKSM_SET_VISIBILITY,
    GKSM_SET_HIGHLIGHTING,
    GKSM_SET_SEGMENT_PRIORITY,
    GKSM_SET_DETECTABILITY,
    GKSM_USER_ITEM			= 101	/* NB: just an indicator; not 
						 * the actual value */
}	gksm_item_id;

/*
 * Suitable for Item type :
 *	0  - END ITEM
 *	2  - REDRAW ALL SEGMENTS ON WORKSTATION
 *	82 - CLOSE SEGMENT
 *
 *	XGKSM0
 */

/*
 * Suitable for Item type :
 *	1  - CLEAR WORKSTAION
 *	3  - UPDATE WORKSTAION
 *	21 - POLYLINE INDEX
 *	22 - LINETYPE
 *	24 - POLYLINE COLOUR INDEX
 *	25 - POLYMARKER INDEX
 *	26 - MARKER TYPE
 *	28 - POLYMARKER COLOUR INDEX
 *	29 - TEXT INDEX
 *	33 - TEXT COLOUR INDEX
 *	35 - TEXT PATH
 *	37 - FILL AREA INDEX
 *	38 - FILL AREA INTERIOR STYLE
 *	39 - FILL AREA STYLE INDEX
 *	40 - FILL AREA COLOUR INDEX
 *	44 - PICK IDENTIFIER
 *	81 - CREATE SEGMENT
 *	84 - DELETE SEGMENT
 */
typedef struct {
    Gint            flag;
}               XGKSMONE;


/*
 * Suitable for Item Type :
 *	4  - DEFERRAL STATE
 *	30 - TEXT FONT AND PRECISION
 *	36 - TEXT ALIGNMENT
 *	83 - RENAME SEGMENT
 *	92 - SET SEGMENT VISIBILITY
 *	93 - SET SEGMENT HIGHLIGHT
 *	95 - SET SEGMENT DETECTABLILITY
 */
typedef struct {
    Gint            item1, item2;
}               XGKSMTWO;


/*
 * Suitable for MESSAGE :
 *	5 - XgksMoMessage
 */
typedef struct {
    Gint            strlen;
    Gchar          *string;
}               XGKSMMESG;


/*
 * Suitable for item type :
 *	11 - POLYLINE
 *	12 - POLYMARKER
 *	14 - FILL AREA
 */
typedef struct {
    Gint            num_pts;
    Gpoint         *pts;
}               XGKSMGRAPH;


/*
 * Suitable for TEXT
 *	13 - XgksMoText
 */
typedef struct {
    Gpoint          location;
    Gint            strlen;
    Gchar          *string;
}               XGKSMTEXT;


/*
 * Suitablr for Cell Array
 *	15 - XgksMoCellArray
 */
typedef struct {
    Gpoint          ll, ur, lr;
    Gipoint         dim;
    Gint           *colour;
}               XGKSMCELLARRAY;


/*
 * Suitable for item type :
 *	23 - LINE WIDTH SCALE FACTOR
 *	27 - MARKER SIZE SCALE FACTOR
 *	31 - CHARACTER EXPANSION FACTOR
 *	32 - CHARACTER SPACING
 */
typedef struct {
    Gfloat          size;
}               XGKSMSIZE;


/*
 * Suitable for CHARACTER VECTRO
 *	34 - XgksMoSetCharVec
 */
typedef struct {
    Gpoint          up, base;
}               XGKSMCHARVEC;


/*
 * Suitable for ASPECT SOURCE FALGS
 *	43 - XgksMoSetAsf
 *
 * There's an extra slot at the end to accomodate the way cgm/cgm.c handles
 * ASF's.
 */
typedef struct {
    Gint            asf[13+1];
}               XGKSMASF;


/*
 * Suitable for item type :
 *	51 - POLYLINE REPRESENTATION
 *	52 - POLYMARKER REPRESENTATION
 */
typedef struct {
    Gint            idx, style, colour;
    Gfloat          size;
}               XGKSMLMREP;


/*
 * Suitable for : TEXT REPRESENTATION
 *	53 - XgksMoSetTextRep
 */
typedef struct {
    Gint            idx, font, prec, colour;
    Gfloat          tx_exp, space;
}               XGKSMTEXTREP;


/*
 * Suitable for FILL AREA REPRESENTATION
 *	54 - XgksMoSetFillRep
 */
typedef struct {
    Gint            idx, intstyle, style, colour;
}               XGKSMFILLREP;


/*
 * Suitable for PATTERN REPRESENTATION
 *	55 - XgksMoSegPatRep
 */
typedef struct {
    Gint            idx;
    Gipoint         size;
    Gint           *array;
}               XGKSMPATREP;


/*
 * Suitable For COLOUR REPRESENTATION
 *	56 - XgksMoSetColourRep
 */
typedef struct {
    Gint            idx;
    Gfloat          red, green, blue;
}               XGKSMCOLOURREP;


/*
 * Suitable for item type :
 *	61 - CLIPPING RECTANGLE
 *	71 - WORKSTATION WINDOW
 *	72 - WORKSTATION VIEWPORT
 */
typedef struct {
    Glimit          rect;
}               XGKSMLIMIT;


/*
 * Suitable for SET SEGMENT TRANSFORMATION
 *	91 - XgksMoSegSegTrans
 */
typedef struct {
    Gint            name;
    Gfloat          matrix[2][3];
}               XGKSMSEGTRAN;


/*
 * Suitable for SET SEGMENT PRIORITY
 *	94 - XgksMoSegSegPri
 */
typedef struct {
    Gint            name;
    Gfloat          pri;
}               XGKSMSEGPRI;


/*
 * Suitable for SET PATTERN SIZE
 *       41 - XgksMoSetPatSiz
 */
typedef struct {
    Gpoint          wid;
    Gpoint          hgt;
}               XGKSMPATSIZ;


/*
 * Suitable for SET PATTERN REFERENCE PT
 *        42 - XgksMoSetPatRef
 */
typedef struct {
    Gpoint          ref;
}               XGKSMPATREF;


/*
 * Metafile API:
 */
extern int XgksMiOpenWs		(WS_STATE_PTR ws);
extern int XgksMoOpenWs		(WS_STATE_PTR ws);
extern int XgksMiCloseWs	(WS_STATE_PTR ws);
extern int XgksMoCloseWs	(WS_STATE_PTR ws, Gint batmode);
extern int XgksMoClearWs	(WS_STATE_PTR ws, Gclrflag flag);
extern int XgksMoReDrawAllSeg	(WS_STATE_PTR ws);
extern int XgksMoUpdateWs	(WS_STATE_PTR ws, Gregen regenflag);
extern int XgksMoDeferWs	(WS_STATE_PTR ws, Gdefmode defer_mode, 
				       Girgmode regen_mode);
extern int XgksMoMessage	(WS_STATE_PTR ws, Gchar *string);
extern int XgksMoGraphicOutputToWs	(WS_STATE_PTR ws, Gint code, 
					       Gint num_pt, Gpoint *pos);
extern int XgksMoGraphicOutput	(Gint code, Gint num_pt, Gpoint *pos);
extern int XgksMoTextToWs	(WS_STATE_PTR ws, Gpoint *at, 
				       Gchar *string);
extern int XgksMoText		(Gpoint *at, Gchar *string);
extern int XgksMoCellArrayToWs	(WS_STATE_PTR ws, Gpoint *ll, Gpoint *ur,
				       Gpoint *lr, 
				       Gint row, Gint *colour, Gipoint *dim);
extern int XgksMoCellArray	(Gpoint *ll, Gpoint *ur, Gpoint *lr,
				       Gint row, Gint *colour, Gipoint *dim);
extern int XgksMoSetGraphicSizeOnWs	(WS_STATE_PTR ws, Gint code, 
					       double size);
extern int XgksMoSetGraphicSize	(Gint code, double size);
extern int XgksMoCloseSegOnWs	(WS_STATE_PTR ws);
extern int XgksMoCloseSeg	(void);
extern int XgksMoSetGraphicAttrOnWs	(WS_STATE_PTR ws, Gint code, 
					      Gint attr);
extern int XgksMoSetGraphicAttr	(Gint code, Gint attr);
extern int XgksMoSetTextFPOnWs	(WS_STATE_PTR ws, Gtxfp *txfp);
extern int XgksMoSetTextFP	(Gtxfp *txfp);
extern int XgksMoSetCharUpOnWs	(WS_STATE_PTR ws, Gpoint *up,
				       Gpoint *base);
extern int XgksMoSetCharUp	(void);
extern int XgksMoSetTextPathOnWs	(WS_STATE_PTR ws, Gtxpath path);
extern int XgksMoSetTextPath	(Gtxpath path);
extern int XgksMoSetTextAlignOnWs	(WS_STATE_PTR ws, 
					      Gtxalign *align);
extern int XgksMoSetTextAlign	(Gtxalign *align);
extern int XgksMoSetFillIntStyleOnWs	(WS_STATE_PTR ws, 
					      Gflinter style);
extern int XgksMoSetFillIntStyle	(Gflinter style);
extern int XgksMoSetPatSizeOnWs	(WS_STATE_PTR ws);
extern int XgksMoSetPatSize	(void);
extern int XgksMoSetPatRefOnWs	(WS_STATE_PTR ws);
extern int XgksMoSetPatRef	(void);
extern int XgksMoSetAsfOnWs	(WS_STATE_PTR ws);
extern int XgksMoSetAsf		(void);
extern int XgksMoSetLineMarkRep	(WS_STATE_PTR ws, Gint code, Gint idx, 
				       Gint type, double size, Gint colour);
extern int XgksMoSetTextRep	(WS_STATE_PTR ws, Gint idx, Gtxbundl *rep);
extern int XgksMoSetFillRep	(WS_STATE_PTR ws, Gint idx, Gflbundl *rep);
extern int XgksMoSetPatRep	(WS_STATE_PTR ws, Gint idx, Gptbundl *rep);
extern int XgksMoSetColourRep	(WS_STATE_PTR ws, Gint idx, Gcobundl *rep);
extern int XgksMoSetClipOnWs	(WS_STATE_PTR ws, Glimit *rect);
extern int XgksMoSetClip	(Glimit *rect);
extern int XgksMoSetLimit	(WS_STATE_PTR ws, Gint code, 
				       Glimit *rect);
extern int XgksMoRenameSeg	(Gint old, Gint new);
extern int XgksMoSetSegTransOnWs	(WS_STATE_PTR ws, Gint name, 
					       Gfloat matrix[2][3]);
extern int XgksMoSetSegTrans	(Gint name, Gfloat matrix[2][3]);
extern int XgksMoSetSegAttrOnWs	(WS_STATE_PTR ws, Gint name, Gint code,
				       Gint attr);
extern int XgksMoSetSegVis	(Gint name, Gsegvis vis);
extern int XgksMoSetSegHiLight	(Gint name, Gseghi hilight);
extern int XgksMoSetSegPriOnWs	(WS_STATE_PTR ws, Gint name, double pri);
extern int XgksMoSetSegPri	(Gint name, double pri);
extern int XgksMoSetSegDet	(Gint name, Gsegdet det);
extern int XgksMoActivateWs	(WS_STATE_PTR ws);
extern int XgksInitGksM		(void);

#endif	/* XGKS_METAFILE_H not defined above */
