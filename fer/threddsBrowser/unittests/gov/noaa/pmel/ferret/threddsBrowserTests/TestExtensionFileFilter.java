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
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import gov.noaa.pmel.ferret.threddsBrowser.ExtensionFileFilter;

import org.junit.Test;

/**
 * Tests of {@link ExtensionFileFilter}
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class TestExtensionFileFilter {

	/**
	 * Test method for {@link ExtensionFileFilter#ExtensionFileFilter()}.
	 * @throws IOException 
	 */
	@Test
	public void testExtensionFileFilter() throws IOException {
		ExtensionFileFilter filter = new ExtensionFileFilter();
		assertNotNull("null returned from empty constructor", filter);
		assertEquals("size of filter set from empty constructor", 0, filter.size());

		// Check that everything gets accepted
		File tmpdir = File.createTempFile("test_", "_dir");
		tmpdir.delete();
		tmpdir.mkdir();
		File afile = new File(tmpdir, "afile.txt");
		afile.createNewFile();
		File bfile = new File(tmpdir, "bfile.dat");
		bfile.createNewFile();
		File cfile = new File(tmpdir, "cfile");
		cfile.createNewFile();
		File subdir = new File(tmpdir, "subdir");
		subdir.mkdir();

		File[] contents = tmpdir.listFiles(filter);
		assertEquals("number of filtered files", 4, contents.length);

		subdir.delete();
		cfile.delete();
		bfile.delete();
		afile.delete();
		tmpdir.delete();
	}

	/**
	 * Test method for {@link ExtensionFileFilter#ExtensionFileFilter(Set)}.
	 * @throws IOException 
	 */
	@Test
	public void testExtensionFileFilterSetOfString() throws IOException {
		List<String> extList = Arrays.asList("txt", "doc", "txt");
		ExtensionFileFilter filter  = new ExtensionFileFilter(extList);
		assertNotNull("null returned from constructor with a List", filter);
		assertEquals("size of filter with List constructor", 2, filter.size());

		// Check that things are filter appropriately
		File tmpdir = File.createTempFile("test_", "_dir");
		tmpdir.delete();
		tmpdir.mkdir();
		File afile = new File(tmpdir, "afile.txt");
		afile.createNewFile();
		File bfile = new File(tmpdir, "bfile.dat");
		bfile.createNewFile();
		File cfile = new File(tmpdir, "cfile");
		cfile.createNewFile();
		File subdir = new File(tmpdir, "subdir");
		subdir.mkdir();

		File[] contents = tmpdir.listFiles(filter);
		assertEquals("number of filtered files", 2, contents.length);

		List<File> contentsList = Arrays.asList(contents);
		assertTrue("file with listed extension accepted", contentsList.contains(afile));
		assertFalse("file with unlisted extension accepted", contentsList.contains(bfile));
		assertFalse("file with no extension accepted", contentsList.contains(cfile));
		assertTrue("directory accepted", contentsList.contains(subdir));

		subdir.delete();
		cfile.delete();
		bfile.delete();
		afile.delete();
		tmpdir.delete();
	}

	/**
	 * Test method for {@link ExtensionFileFilter#accept(File)}.
	 */
	@Test
	public void testAccept() {
		ExtensionFileFilter filter  = new ExtensionFileFilter();
		assertTrue("non-existant file with an extension using filer with no extensions", filter.accept(new File("afile.txt")));
		assertTrue("non-existant file with no extension using filter with no extensions", filter.accept(new File("cfile")));

		List<String> extList = Arrays.asList("txt", "doc");
		filter  = new ExtensionFileFilter(extList);
		assertTrue("file with listed extension", filter.accept(new File("afile.txt")));
		assertFalse("file with unlisted extension", filter.accept(new File("bfile.dat")));
		assertFalse("file with no extension using filter with extensions", filter.accept(new File("cfile")));
	}

}
