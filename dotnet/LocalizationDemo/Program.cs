using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

using SMLMLib;

namespace LocalizationDemo
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            SMLMLib.SMLMLib.SelectNativeLibrary(Directory.GetCurrentDirectory(), false);

            using (var context = new Context())
            {
                Trace.WriteLine($"Context ptr: {context.Instance}");

                var psf = GaussianPSF.CreatePSF_XYIBg(20, 2, 2, true, context);

				float[,] ev = psf.ExpectedValue(new float[,] { { 10, 10, 1000, 2.0f } });

            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
