/**
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
 */
package gov.noaa.pmel.ferret.threddsBrowserTests;

import static org.junit.Assert.*;

import java.io.File;

import gov.noaa.pmel.ferret.threddsBrowser.LocalDirTreeScanMonitor;

import org.junit.Test;

/**
 * Test of {@link LocalDirTreeScanMonitor}
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class TestLocalDirTreeScanMonitor {

	/**
	 * Test method for {@link LocalDirTreeScanMonitor#truncatedPathname} with no parent directory
	 */
	@Test
	public void testTruncatedPathnameNoParent() {
		final String maxTruncName = "..." + File.separator + "pathname";
		final String absFullName = File.separator + "absolute" + File.separator + "testing" + File.separator + "example" + File.separator + "pathname";
		final String partTruncAbsName = File.separator + "absolute" + File.separator + "..." + File.separator + "pathname";
		final String relFullName = "relative" + File.separator + "testing" + File.separator + "example" + File.separator + "pathname";
		final String partTruncRelName = "relative" + File.separator + "..." + File.separator + "pathname";
		final String simpleName = "simple";
		final String shortName = File.separator + "a" + File.separator + "pathname";
		final String complicatedName = File.separator + "an" + File.separator + "extra" + File.separator + "long"    + File.separator + "path"
									 + File.separator + "to" + File.separator + "test"  + File.separator + "against" + File.separator + "pathname";
		final String partTruncComplicatedName = File.separator + "an" + File.separator + "extra" + File.separator + "..." 
									 + File.separator + "against" + File.separator + "pathname";

		// Absolute paths
		File f = new File(absFullName);
		// No truncation
		String name = LocalDirTreeScanMonitor.truncatedPathname(f, null, absFullName.length());
		assertEquals("untruncated absolute pathname", absFullName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, -1);
		assertEquals("max truncated absolute pathname", maxTruncName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, partTruncAbsName.length() + 5);
		assertEquals("partially truncated absolute pathname", partTruncAbsName, name);

		// Relative paths
		f = new File(relFullName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, relFullName.length() + 30);
		assertEquals("untruncated relative pathname", relFullName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, -1);
		assertEquals("max truncated relative pathname", maxTruncName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, partTruncRelName.length());
		assertEquals("partially truncated relative pathname", partTruncRelName, name);

		// Just a name with no path
		f = new File(simpleName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, simpleName.length());
		assertEquals("untruncated simple name", simpleName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, -1);
		assertEquals("max truncated simple name", simpleName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, simpleName.length() - 1);
		assertEquals("partially truncated simple name", simpleName, name);

		// Short pathname
		f = new File(shortName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, shortName.length());
		assertEquals("untruncated short name", shortName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, -1);
		assertEquals("max truncated short name", shortName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, simpleName.length() - 1);
		assertEquals("partially truncated short name", shortName, name);


		// Name with many path components
		f = new File(complicatedName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, complicatedName.length());
		assertEquals("untruncated complicated pathname", complicatedName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, -1);
		assertEquals("max truncated complicated pathname", maxTruncName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, null, partTruncComplicatedName.length() + 1);
		assertEquals("partially truncated complicated pathname", partTruncComplicatedName, name);
	}

	/**
	 * Test method for {@link LocalDirTreeScanMonitor#truncatedPathname} with no parent directory
	 */
	@Test
	public void testTruncatedPathnameWithParent() {
		final String maxTruncName = "..." + File.separator + "pathname";
		final String absParentName = File.separator + "home" + File.separator + "parent";
		final String absFullName = absParentName + File.separator + "absolute" + File.separator + "testing" + File.separator + "example" + File.separator + "pathname";
		final String relAbsFullName = "parent" + File.separator + "absolute" + File.separator + "testing" + File.separator + "example" + File.separator + "pathname";
		final String partTruncAbsName = "parent" + File.separator + "absolute" + File.separator + "..." + File.separator + "pathname";
		final String relParentName = "home" + File.separator + "parent";
		final String relFullName = relParentName + File.separator + "relative" + File.separator + "testing" + File.separator + "example" + File.separator + "pathname";
		final String relRelFullName = "parent" + File.separator + "relative" + File.separator + "testing" + File.separator + "example" + File.separator + "pathname";
		final String partTruncRelName = "parent" + File.separator + "..." + File.separator + "pathname";
		final String simpleName = "simple";
		final String shortName = File.separator + "a" + File.separator + "pathname";

		// Absolute paths
		File p = new File(absParentName);
		File f = new File(absFullName);
		// No truncation
		String name = LocalDirTreeScanMonitor.truncatedPathname(f, p, absFullName.length());
		assertEquals("untruncated absolute pathname", relAbsFullName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, -1);
		assertEquals("max truncated absolute pathname", maxTruncName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, partTruncAbsName.length() + 5);
		assertEquals("partially truncated absolute pathname", partTruncAbsName, name);

		// Relative paths
		p = new File(relParentName);
		f = new File(relFullName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, relFullName.length() + 30);
		assertEquals("untruncated relative pathname", relRelFullName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, -1);
		assertEquals("max truncated relative pathname", maxTruncName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, partTruncRelName.length());
		assertEquals("partially truncated relative pathname", partTruncRelName, name);

		// Just a name with no path
		f = new File(simpleName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, simpleName.length());
		assertEquals("untruncated simple name", simpleName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, -1);
		assertEquals("max truncated simple name", simpleName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, simpleName.length() - 1);
		assertEquals("partially truncated simple name", simpleName, name);

		// Short pathname
		f = new File(shortName);
		// No truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, shortName.length());
		assertEquals("untruncated short name", shortName, name);
		// Maximum truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, -1);
		assertEquals("max truncated short name", shortName, name);
		// Partial truncation
		name = LocalDirTreeScanMonitor.truncatedPathname(f, p, simpleName.length() - 1);
		assertEquals("partially truncated short name", shortName, name);
	}

}
