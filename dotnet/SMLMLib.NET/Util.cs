using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace SMLMLib
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ImageData
    {
        public float* data;
        public int width, height;

        public int Pitch { get { return 4 * width; } }

        public IntPtr Pointer { get { return new IntPtr(data); } }
    }


	public static class UtilExtensionMethods
	{
		public static T MaxElement<T, TCmp>(this IEnumerable<T> seq, Func<T, TCmp> value, T returnOnEmpty = default(T)) where TCmp : IComparable<TCmp>
		{
			T best = returnOnEmpty;
			TCmp vbest = default(TCmp);
			bool none = true;

			foreach (T x in seq)
			{
				TCmp v = value(x);
				if (none || v.CompareTo(vbest) > 0)
				{
					none = false;
					vbest = v;
					best = x;
				}
			}
			return best;
		}
		public static T MinElement<T, TCmp>(this IEnumerable<T> seq, Func<T, TCmp> value, T returnOnEmpty = default(T)) where TCmp : IComparable<TCmp>
		{
			T best = returnOnEmpty;
			TCmp vbest = default(TCmp);
			bool none = true;

			foreach (T x in seq)
			{
				TCmp v = value(x);
				if (none || v.CompareTo(vbest) < 0)
				{
					none = false;
					vbest = v;
					best = x;
				}
			}
			return best;
		}

		public static void ForEach<T>(this IEnumerable<T> seq, Action<T> f)
		{
			foreach (T x in seq) f(x);
		}

		public static int[] Range(this int fromIn, int toEx)
		{
			int[] r = new int[toEx - fromIn];
			for (int i = fromIn; i < toEx; i++)
				r[i] = i;
			return r;
		}
		public static int[] Range(this int len)
		{
			return Range(0, len);
		}

		public static T Last<T>(this List<T> list)
		{
			return list[list.Count - 1];
		}

		public static T Last<T>(this T[] array)
		{
			return array[array.Length - 1];
		}

		public static MinMax<T> GetMinMax<T>(this T[] array) where T : IComparable<T>
		{
			T min = array[0];
			T max = array[0];
			for (int i = 1; i < array.Length; i++)
			{
				if (min.CompareTo(array[i]) > 0) min = array[i];
				if (max.CompareTo(array[i]) < 0) max = array[i];
			}
			return new MinMax<T>() { min = min, max = max };
		}
	}


	public struct MinMax<T>
	{
		public T min, max;
	}


	public class Test
	{
		public static float[] TestArrayPassing(float u, float[] x)
		{
			return Array.ConvertAll(x, a => a * a * u);
		}
		public static void ModifyInPlace(float u, float[] x)
		{
			for (int i = 0; i < x.Length; i++)
				x[i] += u;
		}
		public static void TestByRef(ref int x)
		{
			x = x * 2;
		}

	}

	public class FloatImg : IDisposable
	{
		public IntPtr pixels;
		public int w, h;

		public FloatImg(int w, int h)
		{
			Alloc(w, h);
		}

		public unsafe FloatImg(int w, int h, float* src)
		{
			Alloc(w, h);
			float* dst = (float*)pixels.ToPointer();
			for (int i = 0; i < w * h; i++)
				dst[i] = src[i];
		}

		void Alloc(int w, int h)
		{
			Dispose();
			this.w = w; this.h = h;
			pixels = Marshal.AllocHGlobal(w * h * 4);
		}

		public unsafe FloatImg(Bitmap bmp, byte chan)
		{
			Alloc(bmp.Width, bmp.Height);
			var bmpData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

			float* dst = (float*)pixels.ToPointer();
			var line = bmpData.Scan0;
			for (int y = 0; y < h; y++)
			{
				byte* lp = (byte*)line.ToPointer();

				for (int x = 0; x < w; x++)
					dst[y * w + x] = lp[x * 4 + chan] / 255.0f;

				line = IntPtr.Add(line, bmpData.Stride);
			}
			bmp.UnlockBits(bmpData);
		}

		public unsafe void CopySubimage(FloatImg dstImg, int srcx, int srcy, int dstx, int dsty, int nw, int nh)
		{
			float* src = (float*)pixels.ToPointer();
			float* dst = (float*)dstImg.pixels.ToPointer();
			for (int y = 0; y < nh; y++)
			{
				float* psrc = &src[w * (y + srcy) + srcx];
				float* pdst = &dst[dstImg.w * (y + dsty) + dstx];
				for (int x = 0; x < nw; x++)
					*(pdst++) = *(psrc++);
			}
		}

		public Bitmap ToImage()
		{
			unsafe
			{
				var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
				var bmpData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
				float* src = (float*)pixels.ToPointer();

				var line = bmpData.Scan0;
				for (int y = 0; y < h; y++)
				{
					byte* lp = (byte*)line.ToPointer();

					for (int x = 0; x < w; x++)
					{
						lp[3] = 255;
						lp[0] = lp[1] = lp[2] = (byte)(src[y * w + x] * 255);
						lp += 4;
					}
					line = IntPtr.Add(line, bmpData.Stride);
				}
				bmp.UnlockBits(bmpData);

				return bmp;
			}
		}

		public void Dispose()
		{
			if (pixels != IntPtr.Zero)
			{
				Marshal.FreeHGlobal(pixels);
				pixels = IntPtr.Zero;
			}
		}

		public unsafe ImageData ImageData
		{
			get
			{
				ImageData r = new ImageData()
				{
					data = (float*)pixels.ToPointer(),
					width = w,
					height = h
				};
				return r;
			}

		}

		[DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
		static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

		public FloatImg ExtractSubsection(int y, int nh)
		{
			IntPtr start = IntPtr.Add(pixels, y * 4 * w);
			FloatImg dst = new FloatImg(w, nh);
			CopyMemory(dst.pixels, start, (uint)(4 * nh * w));
			return dst;
		}
	}	
}
