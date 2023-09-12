using SMLMLib;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;
using System.Runtime.InteropServices;

using PythonPlotter;

namespace LocalizerTest
{
	public partial class Form1 : Form
	{
		Vector2 estimatedPos;
		

		public Form1()
		{
			InitializeComponent();
		}

		private void button1_Click(object sender, EventArgs e)
		{


		}

		private void pictureSample_Click(object sender, EventArgs e)
		{
			
		}

		private void pictureSample_MouseDown(object sender, MouseEventArgs e)
		{
			Resample(e.Location);
		}

		private void pictureSample_MouseMove(object sender, MouseEventArgs e)
		{
			if(e.Button == MouseButtons.Left)
				Resample(e.Location);
		}

		Bitmap FloatToBitmap(float[,] img)
		{
			var bmp = new Bitmap(img.GetLength(1), img.GetLength(0));

			float min, max;
			min = max = img[0, 0];

			foreach (var f in img)
			{
				if (min > f) min = f;
				if (max < f) max = f;
			}

			for (int y = 0; y < bmp.Height; y++)
				for (int x = 0; x < bmp.Width; x++)
				{
					int v = (int)(255 * (img[y, x] - min) / (max - min));
					if (v > 255) v = 255;
					if (v < 0) v = 0;
					bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
				}
			return bmp;
		}

		void Resample(Point pt)
		{
//			pictureSample.Image = bmp;
		}

		private void pictureSample_Paint(object sender, PaintEventArgs e)
		{
			e.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
			var img = pictureSample.Image;
			if (img != null)
			{
				e.Graphics.DrawImage(
					img,
					new Rectangle(0, 0, pictureSample.Width, pictureSample.Height),
					// destination rectangle 
					-0.5f,
					-0.5f,           // upper-left corner of source rectangle
					img.Width+0.5f,       // width of source rectangle
					img.Height+0.5f,      // height of source rectangle
					GraphicsUnit.Pixel);

				var x = estimatedPos.x / img.Width * pictureSample.Width;
				var y = estimatedPos.y / img.Height * pictureSample.Height;
				e.Graphics.DrawEllipse(Pens.Blue, x, y, 5, 5);
			}
		}
		
	}

}
