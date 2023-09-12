using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Threading;
using SMLMLib;

namespace TiffViewer
{

	public partial class MainDialog : Form
	{
		TiffReader reader;
		List<ushort[]> images;
		List<XYZIBg[]> results=new List<XYZIBg[]>();

		List<ILocalizationMethod> localizationMethods = new List<ILocalizationMethod>();

		public MainDialog()
		{
			InitializeComponent();

			foreach (var t in FindImplementations.Find(typeof(ILocalizationMethod)))
			{
				var lm = (ILocalizationMethod)Activator.CreateInstance(t);
				comboBoxMethods.Items.Add(lm);
				localizationMethods.Add(lm);
			}
			if (localizationMethods.Count > 0)
				comboBoxMethods.SelectedIndex = 0;
		}


		unsafe Bitmap ConvertToBitmap(ushort[] data,bool autoscale)
		{
			Size imageSize = reader.ImageSize;
			int bps = reader.BitsPerPixel;
			var bmp = new Bitmap(imageSize.Width, imageSize.Height, PixelFormat.Format32bppRgb);

			var minmax = data.GetMinMax();
			int offset = minmax.min;
			float scale =255.0f/(minmax.max - minmax.min);

			var bits = bmp.LockBits(new Rectangle(new Point(), imageSize), ImageLockMode.WriteOnly, PixelFormat.Format32bppRgb);
			for (int y=0;y<imageSize.Height;y++)
			{
				uint* dst = (uint*)(IntPtr.Add(bits.Scan0, bits.Stride * y).ToPointer());
				fixed (ushort* src = data) {
					for (int x = 0; x < imageSize.Width; x++)
					{
						int srcval = src[y * imageSize.Width + x];
						int col;
						if (autoscale)
							col = (int)((srcval - offset) * scale);
						else
							col=(int)(srcval >> (bps - 8));

						uint v = (uint)((col << 16) + (col << 8) + col);
						dst[x] = v;
					}
				}
			}
			bmp.UnlockBits(bits);
			return bmp;
		}

		private void openTIFFToolStripMenuItem_Click(object sender, EventArgs e)
		{
			var ofd = new OpenFileDialog()
			{
				Filter = "TIFF files (*.tiff,*.tif)|*.tiff;*.tif"
			};

			if (ofd.ShowDialog() == DialogResult.OK)
			{
				if (reader != null)
					reader.Dispose();

				using (reader = new TiffReader(ofd.FileName))
				{
					int framenum = 0;
					ushort[] img;
					while ((img = reader.ReadNextImage()) != null)
					{
						var minmax = img.GetMinMax();
						if (framenum % 100 == 0)
							Trace.WriteLine($"Read frame {framenum}. Min={minmax.min}, Max={minmax.max}");

						if (framenum == 0)
						{
							pictureBox.Image = ConvertToBitmap(img, true);
						}
						framenum++;
					}
					trackBarFrame.Minimum = 0;
					trackBarFrame.Maximum = framenum - 1;
					trackBarFrame.Value = 0;
				}
			}
		}

		private void trackBarFrame_Scroll(object sender, EventArgs e)
		{
			pictureBox.Image = ConvertToBitmap(images[trackBarFrame.Value],true);
			pictureBox.Invalidate();
		}

		private void comboBoxMethods_SelectedIndexChanged(object sender, EventArgs e)
		{
			configPropertyGrid.SelectedObject = localizationMethods[comboBoxMethods.SelectedIndex].GetConfig();
			configPropertyGrid.ExpandAllGridItems();
		}

		private void buttonProcess_Click(object sender, EventArgs e)
		{
			if (reader == null)
				return;

			var lm = localizationMethods[comboBoxMethods.SelectedIndex];
			using (ImageQueue queue = lm.CreateImageQueue(reader.ImageSize, configPropertyGrid.SelectedObject)) {
				queue.Start();

				for (int i = 0; i < images.Count; i++)
				{
					queue.PushFrame(i, images[i]);
				}

				while (!queue.IsIdle())
					Thread.Sleep(10);

				int fc = queue.GetFrameCount();
				results.Clear();
				for (int f=0;f<fc;f++)
				{
					//					int rc = queue.GetFrameResultCount(f);
					XYZIBg[] pts = queue.GetFrameResults(f);
					Trace.WriteLine($"{pts.Length} results for frame {f}");

					results.Add(pts);
				}
			}
		}

		private void pictureBox_Paint(object sender, PaintEventArgs e)
		{
			int w = pictureBox.Width, h = pictureBox.Height;

			int frame = trackBarFrame.Value;
			if (frame >= 0 && frame < results.Count)
			{
				Size s = reader.ImageSize;
				float xscale = w / (float)s.Width;
				float yscale = h / (float)s.Height;

				float R = 5.0f;
				foreach (var r in results[frame])
				{
					e.Graphics.DrawEllipse(Pens.Blue, r.x * xscale-R, r.y * yscale-R, R*2, R*2);
				}
			}
		}
	}
}
