using System;

using System.Diagnostics;
using System.Drawing;


using BitMiracle.LibTiff.Classic;



public class TiffReader : IDisposable
{
	Tiff tiff;
	int w, h, channels;
	int numImages;
	int bytesPerPixel, bitsPerPixel;

	public int BitsPerPixel
	{
		get { return bitsPerPixel; }
	}

	public Size ImageSize
	{
		get { return new Size(w, h); }
	}

	public TiffReader(string fn)
	{
		tiff = Tiff.Open(fn, "r");

		w = tiff.GetField(TiffTag.IMAGEWIDTH)[0].ToInt();
		h = tiff.GetField(TiffTag.IMAGELENGTH)[0].ToInt();
		bitsPerPixel = tiff.GetField(TiffTag.BITSPERSAMPLE)[0].ToInt();
		bytesPerPixel = (bitsPerPixel + 7) / 8;
		channels = tiff.GetField(TiffTag.SAMPLESPERPIXEL)[0].ToInt();

		numImages = tiff.NumberOfDirectories();

		Debug.Assert(channels == 1);
	}

	public int NumImages
	{
		get { return numImages;}
	}


	public ushort[] ReadFrame(int frame)
	{
		if (frame < 0 || frame >= NumImages)
			throw new ApplicationException($"Invalid frame number {frame}");

		tiff.SetDirectory((short)frame);

		ushort[] image = new ushort[w * h];
		byte[] buffer = new byte[bytesPerPixel * w];
		for (int y = 0; y < h; y++)
		{
			tiff.ReadScanline(buffer, y);

			if (bytesPerPixel == 1)
			{
				for (int x = 0; x < w; x++)
					image[y * w + x] = buffer[x];
			}
			else
			{
				for (int x = 0; x < w; x++)
				{
					ushort v = (ushort)(buffer[x * 2+1] * 256U + buffer[x * 2]);
					image[y * w + x] = v;
				}
			}
		}
		return image;
	}

	#region IDisposable Support
	private bool disposedValue = false; // To detect redundant calls

	protected virtual void Dispose(bool disposing)
	{
		if (!disposedValue)
		{
			if (disposing)
				tiff.Dispose();
			disposedValue = true;
		}
	}

	// TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
	// ~TiffReader() {
	//   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
	//   Dispose(false);
	// }

	// This code added to correctly implement the disposable pattern.
	public void Dispose()
	{
		// Do not change this code. Put cleanup code in Dispose(bool disposing) above.
		Dispose(true);
		// TODO: uncomment the following line if the finalizer is overridden above.
		// GC.SuppressFinalize(this);
	}
	#endregion

}