namespace LocalizerTest
{
	partial class Form1
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.plotView1 = new OxyPlot.WindowsForms.PlotView();
			this.buttonTestMLE = new System.Windows.Forms.Button();
			this.pictureSample = new System.Windows.Forms.PictureBox();
			this.labelCRLB = new System.Windows.Forms.Label();
			this.labelMLE = new System.Windows.Forms.Label();
			((System.ComponentModel.ISupportInitialize)(this.pictureSample)).BeginInit();
			this.SuspendLayout();
			// 
			// plotView1
			// 
			this.plotView1.Location = new System.Drawing.Point(354, 12);
			this.plotView1.Name = "plotView1";
			this.plotView1.PanCursor = System.Windows.Forms.Cursors.Hand;
			this.plotView1.Size = new System.Drawing.Size(434, 259);
			this.plotView1.TabIndex = 1;
			this.plotView1.Text = "plotView1";
			this.plotView1.ZoomHorizontalCursor = System.Windows.Forms.Cursors.SizeWE;
			this.plotView1.ZoomRectangleCursor = System.Windows.Forms.Cursors.SizeNWSE;
			this.plotView1.ZoomVerticalCursor = System.Windows.Forms.Cursors.SizeNS;
			// 
			// buttonTestMLE
			// 
			this.buttonTestMLE.Location = new System.Drawing.Point(40, 305);
			this.buttonTestMLE.Name = "buttonTestMLE";
			this.buttonTestMLE.Size = new System.Drawing.Size(75, 23);
			this.buttonTestMLE.TabIndex = 2;
			this.buttonTestMLE.Text = "button1";
			this.buttonTestMLE.UseVisualStyleBackColor = true;
			this.buttonTestMLE.Click += new System.EventHandler(this.button1_Click);
			// 
			// pictureSample
			// 
			this.pictureSample.Location = new System.Drawing.Point(12, 12);
			this.pictureSample.Name = "pictureSample";
			this.pictureSample.Size = new System.Drawing.Size(311, 274);
			this.pictureSample.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
			this.pictureSample.TabIndex = 3;
			this.pictureSample.TabStop = false;
			this.pictureSample.Click += new System.EventHandler(this.pictureSample_Click);
			this.pictureSample.Paint += new System.Windows.Forms.PaintEventHandler(this.pictureSample_Paint);
			this.pictureSample.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pictureSample_MouseDown);
			this.pictureSample.MouseMove += new System.Windows.Forms.MouseEventHandler(this.pictureSample_MouseMove);
			// 
			// labelCRLB
			// 
			this.labelCRLB.AutoSize = true;
			this.labelCRLB.Location = new System.Drawing.Point(12, 354);
			this.labelCRLB.Name = "labelCRLB";
			this.labelCRLB.Size = new System.Drawing.Size(57, 13);
			this.labelCRLB.TabIndex = 4;
			this.labelCRLB.Text = "labelCRLB";
			// 
			// labelMLE
			// 
			this.labelMLE.AutoSize = true;
			this.labelMLE.Location = new System.Drawing.Point(12, 387);
			this.labelMLE.Name = "labelMLE";
			this.labelMLE.Size = new System.Drawing.Size(29, 13);
			this.labelMLE.TabIndex = 4;
			this.labelMLE.Text = "MLE";
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(800, 450);
			this.Controls.Add(this.labelMLE);
			this.Controls.Add(this.labelCRLB);
			this.Controls.Add(this.pictureSample);
			this.Controls.Add(this.buttonTestMLE);
			this.Controls.Add(this.plotView1);
			this.Name = "Form1";
			this.Text = "Form1";
			((System.ComponentModel.ISupportInitialize)(this.pictureSample)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private OxyPlot.WindowsForms.PlotView plotView1;
		private System.Windows.Forms.Button buttonTestMLE;
		private System.Windows.Forms.PictureBox pictureSample;
		private System.Windows.Forms.Label labelCRLB;
		private System.Windows.Forms.Label labelMLE;
	}
}

