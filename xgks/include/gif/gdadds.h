#ifndef GIFADDS_H
#define GIFADDS_h
void gdImageWideLine(gdImagePtr im, int x1, int y1, int x2, int y2,
		     int color, int width);
void gdImageWideLines(gdImagePtr im, gdPointPtr pts, int num,
		     int color, int width);
void gdImageBlockFill(gdImagePtr image, int color);
#endif
